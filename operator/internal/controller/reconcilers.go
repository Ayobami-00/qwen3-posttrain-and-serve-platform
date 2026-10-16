package controller

import (
	"context"
	"errors"
	"fmt"
	cp "github.com/Ayobami-00/qwen3-posttrain-and-serve-platform/operator/internal/controlplane"
	"github.com/Ayobami-00/qwen3-posttrain-and-serve-platform/operator/internal/provider"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"time"
)

const finalizer = "qps.dev/cleanup"

var interval = 5 * time.Second

func terminal(value string) bool {
	return value == "succeeded" || value == "failed" || value == "cancelled" || value == "interrupted"
}

type WorkloadReconciler struct {
	client.Client
	API      *cp.Client
	Provider *provider.Runpod
	Kind     string
}

func (r *WorkloadReconciler) Reconcile(ctx context.Context, request ctrl.Request) (ctrl.Result, error) {
	var object client.Object
	var spec *WorkloadSpec
	var status *WorkloadStatus
	if r.Kind == "training" {
		v := &TrainingRun{}
		object = v
		spec = &v.Spec
		status = &v.Status
	} else {
		v := &ModelDeployment{}
		object = v
		spec = &v.Spec
		status = &v.Status
	}
	if err := r.Get(ctx, request.NamespacedName, object); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	owner, err := r.API.Workload(ctx, r.Kind, spec.OwnerID)
	if err != nil {
		return ctrl.Result{RequeueAfter: interval}, err
	}
	revision := owner.Revision(r.Kind)
	if revision != spec.Generation {
		return ctrl.Result{RequeueAfter: interval}, nil
	}
	var allocation *cp.Allocation
	for _, a := range owner.Allocations {
		if a.Generation == revision {
			copy := a
			allocation = &copy
			break
		}
	}
	phase := owner.Status
	message := ""
	if allocation == nil {
		if owner.Desired != "run" || !object.GetDeletionTimestamp().IsZero() {
			phase = "cancelled"
		} else if !terminal(owner.Status) {
			quote := provider.Quote{HourlyRate: 0, GPUType: "cpu"}
			if owner.Profile(r.Kind) != "cpu-fixture" {
				quote, err = r.Provider.Quote(ctx)
				if err != nil {
					return ctrl.Result{RequeueAfter: 30 * time.Second}, nil
				}
			}
			created, reserveErr := r.API.Reserve(ctx, r.Kind, owner.ID, quote)
			if reserveErr != nil {
				var httpErr cp.HTTPError
				if errors.As(reserveErr, &httpErr) && httpErr.Status == 409 {
					return ctrl.Result{RequeueAfter: interval}, nil
				}
				return ctrl.Result{}, reserveErr
			}
			allocation = &created
			phase = "provisioning"
		}
	}
	if allocation != nil {
		if allocation.State == "terminated" {
			if owner.Desired != "run" {
				phase = "cancelled"
			} else if allocation.Result != nil {
				phase = allocation.Result.Status
				message = allocation.Result.Reason
			} else {
				phase = "interrupted"
				message = "Compute ended before a complete worker result"
				if r.Kind == "training" && owner.Attempt == 1 && time.Now().Unix() < int64(allocation.Deadline) {
					if recoverErr := r.API.Request(ctx, "POST", "/workloads/training/"+owner.ID+"/recover", nil, nil); recoverErr == nil {
						return ctrl.Result{RequeueAfter: interval}, nil
					}
				}
			}
		} else {
			child := &ComputeAllocation{ObjectMeta: metav1.ObjectMeta{Name: allocation.ID, Namespace: request.Namespace},
				Spec: AllocationSpec{AllocationID: allocation.ID, OwnerID: owner.ID, OwnerKind: r.Kind, Generation: revision}}
			if err := controllerutil.SetControllerReference(object, child, r.Scheme()); err != nil {
				return ctrl.Result{}, err
			}
			if err := r.Create(ctx, child); err != nil && !apierrors.IsAlreadyExists(err) {
				return ctrl.Result{}, err
			}
			phase = "provisioning"
			if allocation.Heartbeat != nil && (r.Kind == "training" || allocation.Ready) {
				phase = "running"
			}
			if owner.Desired != "run" {
				phase = "cancelling"
			}
		}
	}
	status.Phase = phase
	status.ObservedGeneration = object.GetGeneration()
	status.Message = message
	if allocation != nil {
		status.AllocationID = allocation.ID
	}
	if err := r.Status().Update(ctx, object); err != nil {
		return ctrl.Result{}, err
	}
	detail := phase
	if allocation != nil && allocation.Phase != "" {
		detail = allocation.Phase
	}
	if err := r.API.Project(ctx, r.Kind, owner.ID, phase, detail, revision); err != nil {
		return ctrl.Result{}, err
	}
	return ctrl.Result{RequeueAfter: interval}, nil
}

type AllocationReconciler struct {
	client.Client
	API      *cp.Client
	Provider *provider.Runpod
}

func (r *AllocationReconciler) Reconcile(ctx context.Context, request ctrl.Request) (ctrl.Result, error) {
	var resource ComputeAllocation
	if err := r.Get(ctx, request.NamespacedName, &resource); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	if !controllerutil.ContainsFinalizer(&resource, finalizer) {
		controllerutil.AddFinalizer(&resource, finalizer)
		return ctrl.Result{}, r.Update(ctx, &resource)
	}
	plan, err := r.API.Allocation(ctx, resource.Spec.AllocationID)
	if err != nil {
		return ctrl.Result{RequeueAfter: interval}, err
	}
	if plan.State == "terminated" {
		if !resource.DeletionTimestamp.IsZero() {
			controllerutil.RemoveFinalizer(&resource, finalizer)
			return ctrl.Result{}, r.Update(ctx, &resource)
		}
		return ctrl.Result{}, nil
	}
	expired := float64(time.Now().Unix()) >= plan.Deadline || float64(time.Now().Unix()) >= plan.LeaseUntil
	stopping := expired || plan.Desired != "run" || plan.OwnerDesired != "run" || plan.Result != nil || !resource.DeletionTimestamp.IsZero()
	cpu := &CPUProvider{Client: r.Client}
	if stopping {
		if plan.Profile == "cpu-fixture" {
			if err := cpu.Delete(ctx, plan); err != nil {
				return ctrl.Result{}, err
			}
			if _, err := cpu.Get(ctx, plan); !errors.Is(err, provider.ErrNotFound) {
				return ctrl.Result{RequeueAfter: interval}, err
			}
		} else {
			pods, err := r.Provider.List(ctx)
			if err != nil {
				return ctrl.Result{}, err
			}
			found := false
			for _, pod := range pods {
				if pod.Name == plan.ResourceName {
					if resource.Status.PodID != "" && pod.ID != resource.Status.PodID {
						return ctrl.Result{}, errors.New("provider identity conflict; cleanup requires reconciliation")
					}
					found = true
					if err := r.Provider.Delete(ctx, pod.ID); err != nil {
						return ctrl.Result{}, err
					}
				}
			}
			if found {
				return ctrl.Result{RequeueAfter: interval}, nil
			}
			if resource.Status.PodID != "" {
				if _, err := r.Provider.Get(ctx, resource.Status.PodID); err == nil {
					return ctrl.Result{RequeueAfter: interval}, nil
				} else if !errors.Is(err, provider.ErrNotFound) {
					return ctrl.Result{}, err
				}
			}
			if err := r.Provider.DeleteRegistry(ctx, resource.Status.RegistryID); err != nil {
				return ctrl.Result{}, err
			}
		}
		if _, err := r.API.Observe(ctx, plan.ID, map[string]any{"state": "terminated"}); err != nil {
			return ctrl.Result{}, err
		}
		resource.Status.Phase = "terminated"
		return ctrl.Result{}, r.Status().Update(ctx, &resource)
	}
	if !resource.Status.CreateRequested {
		if plan.Profile != "cpu-fixture" {
			registry, err := r.Provider.EnsureRegistry(ctx, plan)
			if err != nil {
				return ctrl.Result{RequeueAfter: interval}, err
			}
			resource.Status.RegistryID = registry
		}
		resource.Status.CreateRequested = true
		resource.Status.RequestedAt = time.Now().Unix()
		resource.Status.Phase = "creating"
		if err := r.Status().Update(ctx, &resource); err != nil {
			return ctrl.Result{}, err
		}
		if _, err := r.API.Observe(ctx, plan.ID, map[string]any{"state": "creating"}); err != nil {
			return ctrl.Result{}, err
		}
		var pod provider.Pod
		if plan.Profile == "cpu-fixture" {
			pod, err = cpu.Create(ctx, plan)
		} else {
			pod, err = r.Provider.Create(ctx, plan, resource.Status.RegistryID)
		}
		if err != nil {
			return ctrl.Result{RequeueAfter: interval}, nil
		}
		resource.Status.PodID = pod.ID
		if err := r.Status().Update(ctx, &resource); err != nil {
			return ctrl.Result{}, err
		}
	}
	var pod provider.Pod
	if plan.Profile == "cpu-fixture" {
		pod, err = cpu.Get(ctx, plan)
	} else if resource.Status.PodID != "" {
		pod, err = r.Provider.Get(ctx, resource.Status.PodID)
	} else {
		pods, listErr := r.Provider.List(ctx)
		if listErr != nil {
			return ctrl.Result{}, listErr
		}
		matches := []provider.Pod{}
		for _, v := range pods {
			if v.Name == plan.ResourceName {
				matches = append(matches, v)
			}
		}
		if len(matches) > 1 {
			_, _ = r.API.Observe(ctx, plan.ID, map[string]any{"desired": "stop"})
			return ctrl.Result{RequeueAfter: interval}, nil
		}
		if len(matches) == 0 {
			return ctrl.Result{RequeueAfter: interval}, nil
		}
		pod = matches[0]
		resource.Status.PodID = pod.ID
		if err := r.Status().Update(ctx, &resource); err != nil {
			return ctrl.Result{}, err
		}
	}
	if errors.Is(err, provider.ErrNotFound) {
		_, err = r.API.Observe(ctx, plan.ID, map[string]any{"desired": "stop"})
		return ctrl.Result{RequeueAfter: interval}, err
	}
	if err != nil {
		return ctrl.Result{}, err
	}
	if pod.Name != plan.ResourceName || pod.Image != plan.Image {
		return ctrl.Result{}, errors.New("provider allocation identity mismatch")
	}
	if plan.Profile != "cpu-fixture" && (pod.Cost <= 0 || pod.Cost > plan.HourlyRate+0.000001 || pod.Cost > 1) {
		_, err = r.API.Observe(ctx, plan.ID, map[string]any{"desired": "stop"})
		return ctrl.Result{RequeueAfter: interval}, err
	}
	if pod.Status == "EXITED" || pod.Status == "FAILED" || pod.Status == "STOPPED" {
		_, err = r.API.Observe(ctx, plan.ID, map[string]any{"desired": "stop"})
		return ctrl.Result{RequeueAfter: interval}, err
	}
	endpoint := fmt.Sprintf("https://%s-8000.proxy.runpod.net", pod.ID)
	if plan.Profile == "cpu-fixture" {
		endpoint = CPUURL(plan)
	}
	observed := map[string]any{"pod_id": pod.ID, "state": "running", "endpoint": endpoint}
	if _, err := r.API.Observe(ctx, plan.ID, observed); err != nil {
		return ctrl.Result{}, err
	}
	resource.Status.Phase = "running"
	if err := r.Status().Update(ctx, &resource); err != nil {
		return ctrl.Result{}, err
	}
	return ctrl.Result{RequeueAfter: interval}, nil
}

func (r *WorkloadReconciler) SetupWithManager(manager ctrl.Manager) error {
	if r.Kind == "training" {
		return ctrl.NewControllerManagedBy(manager).For(&TrainingRun{}).Owns(&ComputeAllocation{}).Complete(r)
	}
	return ctrl.NewControllerManagedBy(manager).For(&ModelDeployment{}).Owns(&ComputeAllocation{}).Complete(r)
}
func (r *AllocationReconciler) SetupWithManager(manager ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(manager).For(&ComputeAllocation{}).Complete(r)
}
