package controller

import (
	"context"
	"errors"
	"fmt"
	cp "github.com/Ayobami-00/qwen3-posttrain-and-serve-platform/operator/internal/controlplane"
	"github.com/Ayobami-00/qwen3-posttrain-and-serve-platform/operator/internal/provider"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

type CPUProvider struct{ Client client.Client }

func (p *CPUProvider) Create(ctx context.Context, a cp.Allocation) (provider.Pod, error) {
	labels := map[string]string{"qps.dev/allocation": a.ID, "app.kubernetes.io/part-of": "qps"}
	env := []corev1.EnvVar{{Name: "QPS_API_URL", Value: a.APIURL}, {Name: "QPS_ALLOCATION_ID", Value: a.ID}, {Name: "QPS_WORKER_TOKEN", Value: a.WorkerToken}, {Name: "QPS_IMAGE_DIGEST", Value: a.Image}, {Name: "QPS_WORK_DIR", Value: "/workspace/qps"}}
	job := &batchv1.Job{ObjectMeta: metav1.ObjectMeta{Name: a.ResourceName, Namespace: a.Namespace, Labels: labels}, Spec: batchv1.JobSpec{BackoffLimit: ptr.To[int32](0), ActiveDeadlineSeconds: ptr.To[int64](int64(a.Deadline - a.Created)), Template: corev1.PodTemplateSpec{ObjectMeta: metav1.ObjectMeta{Labels: labels}, Spec: corev1.PodSpec{RestartPolicy: corev1.RestartPolicyNever, AutomountServiceAccountToken: ptr.To(false), Containers: []corev1.Container{{Name: "worker", Image: a.Image, Command: []string{"python", "-m", "qps.hosted.agent"}, Resources: corev1.ResourceRequirements{Requests: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("250m"), corev1.ResourceMemory: resource.MustParse("512Mi")}, Limits: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("1"), corev1.ResourceMemory: resource.MustParse("1536Mi")}}, Env: env, Ports: []corev1.ContainerPort{{ContainerPort: 8000}}}}}}}}
	if err := p.Client.Create(ctx, job); err != nil && !apierrors.IsAlreadyExists(err) {
		return provider.Pod{}, err
	}
	if a.Kind == "deployment" {
		svc := &corev1.Service{ObjectMeta: metav1.ObjectMeta{Name: a.ResourceName, Namespace: a.Namespace, Labels: labels}, Spec: corev1.ServiceSpec{Selector: labels, Ports: []corev1.ServicePort{{Port: 8000, TargetPort: intstr.FromInt32(8000)}}}}
		if err := p.Client.Create(ctx, svc); err != nil && !apierrors.IsAlreadyExists(err) {
			return provider.Pod{}, err
		}
	}
	return provider.Pod{ID: a.ResourceName, Name: a.ResourceName, Status: "PENDING", Cost: 0, Image: a.Image}, nil
}
func (p *CPUProvider) Get(ctx context.Context, a cp.Allocation) (provider.Pod, error) {
	var job batchv1.Job
	if err := p.Client.Get(ctx, types.NamespacedName{Name: a.ResourceName, Namespace: a.Namespace}, &job); err != nil {
		if apierrors.IsNotFound(err) {
			return provider.Pod{}, provider.ErrNotFound
		}
		return provider.Pod{}, err
	}
	status := "PENDING"
	if job.Status.Active > 0 {
		status = "RUNNING"
	}
	if job.Status.Succeeded > 0 || job.Status.Failed > 0 {
		status = "EXITED"
	}
	return provider.Pod{ID: a.ResourceName, Name: a.ResourceName, Status: status, Image: a.Image}, nil
}
func (p *CPUProvider) Delete(ctx context.Context, a cp.Allocation) error {
	var job batchv1.Job
	err := p.Client.Get(ctx, types.NamespacedName{Name: a.ResourceName, Namespace: a.Namespace}, &job)
	if err == nil {
		if job.Labels["qps.dev/allocation"] != a.ID {
			return errors.New("refusing to delete unowned CPU job")
		}
		policy := metav1.DeletePropagationForeground
		err = p.Client.Delete(ctx, &job, &client.DeleteOptions{PropagationPolicy: &policy})
	}
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	svc := &corev1.Service{ObjectMeta: metav1.ObjectMeta{Name: a.ResourceName, Namespace: a.Namespace}}
	if err := p.Client.Get(ctx, types.NamespacedName{Name: a.ResourceName, Namespace: a.Namespace}, svc); err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return err
	}
	if svc.Labels["qps.dev/allocation"] != a.ID {
		return errors.New("refusing to delete unowned CPU service")
	}
	if err := p.Client.Delete(ctx, svc); err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	return nil
}
func CPUURL(a cp.Allocation) string {
	return fmt.Sprintf("http://%s.%s.svc:8000", a.ResourceName, a.Namespace)
}
