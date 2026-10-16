package controller

import (
	"context"
	"encoding/json"
	cp "github.com/Ayobami-00/qwen3-posttrain-and-serve-platform/operator/internal/controlplane"
	"github.com/Ayobami-00/qwen3-posttrain-and-serve-platform/operator/internal/provider"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"net/http"
	"net/http/httptest"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestUncertainCreateIsReconciledOnceAcrossRestart(t *testing.T) {
	var mu sync.Mutex
	plan := cp.Allocation{ID: "alloc-01234567890123456789012345678901", OwnerID: "run-01234567890123456789012345678901", Kind: "training", Profile: "runpod-qwen3-small", State: "reserved", Desired: "run", OwnerDesired: "run", Generation: 1, HourlyRate: .59, Image: "example@sha256:abc", ResourceName: "qps-test-owned", Created: float64(time.Now().Unix()), Deadline: float64(time.Now().Unix() + 2700), LeaseUntil: float64(time.Now().Unix() + 600)}
	control := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		if strings.HasSuffix(r.URL.Path, "/observe") {
			var body map[string]any
			_ = json.NewDecoder(r.Body).Decode(&body)
			if value, ok := body["state"].(string); ok {
				plan.State = value
			}
			if value, ok := body["desired"].(string); ok {
				plan.Desired = value
			}
			if value, ok := body["pod_id"].(string); ok {
				plan.PodID = value
			}
		}
		_ = json.NewEncoder(w).Encode(plan)
	}))
	defer control.Close()
	creates := 0
	deletes := []string{}
	pods := map[string]provider.Pod{"foreign": {ID: "foreign", Name: "another-project", Image: "other"}}
	cloud := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == "POST" && r.URL.Path == "/pods":
			creates++
			pods["owned"] = provider.Pod{ID: "owned", Name: plan.ResourceName, Image: plan.Image, Cost: .59, Status: "RUNNING"}
			w.WriteHeader(503) // Provider committed, response was lost.
		case r.Method == "GET" && r.URL.Path == "/pods":
			values := []provider.Pod{}
			for _, p := range pods {
				values = append(values, p)
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"pods": values})
		case r.Method == "DELETE":
			id := strings.TrimPrefix(r.URL.Path, "/pods/")
			deletes = append(deletes, id)
			delete(pods, id)
			w.WriteHeader(204)
		default:
			id := strings.TrimPrefix(r.URL.Path, "/pods/")
			if p, ok := pods[id]; ok {
				_ = json.NewEncoder(w).Encode(p)
			} else {
				w.WriteHeader(404)
			}
		}
	}))
	defer cloud.Close()
	scheme := runtime.NewScheme()
	_ = AddToScheme(scheme)
	resource := &ComputeAllocation{ObjectMeta: metav1.ObjectMeta{Name: plan.ID, Namespace: "qps"}, Spec: AllocationSpec{AllocationID: plan.ID}}
	client := fake.NewClientBuilder().WithScheme(scheme).WithStatusSubresource(&ComputeAllocation{}).WithObjects(resource).Build()
	api := &cp.Client{URL: control.URL, Token: strings.Repeat("i", 32)}
	providerAPI := &provider.Runpod{URL: cloud.URL}
	request := ctrl.Request{NamespacedName: types.NamespacedName{Name: plan.ID, Namespace: "qps"}}
	for i := 0; i < 5; i++ {
		// Reconstruct the reconciler to exercise durable status rather than process memory.
		reconciler := AllocationReconciler{Client: client, API: api, Provider: providerAPI}
		if _, err := reconciler.Reconcile(context.Background(), request); err != nil {
			t.Fatal(err)
		}
	}
	if creates != 1 || plan.PodID != "owned" {
		t.Fatalf("creation was not reconciled: calls=%d plan=%+v", creates, plan)
	}
	plan.Desired = "stop"
	for i := 0; i < 3; i++ {
		reconciler := AllocationReconciler{Client: client, API: api, Provider: providerAPI}
		if _, err := reconciler.Reconcile(context.Background(), request); err != nil {
			t.Fatal(err)
		}
	}
	if plan.State != "terminated" {
		t.Fatal("termination was not confirmed")
	}
	if len(deletes) != 1 || deletes[0] != "owned" {
		t.Fatalf("deleted unowned resources: %v", deletes)
	}
	if _, ok := pods["foreign"]; !ok {
		t.Fatal("foreign pod was removed")
	}
}
