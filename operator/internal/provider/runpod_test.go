package provider

import (
	"context"
	"encoding/json"
	cp "github.com/Ayobami-00/qwen3-posttrain-and-serve-platform/operator/internal/controlplane"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestQuotePrefersValidatedAffordableCapacity(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer private-key" {
			t.Error("missing provider authentication")
		}
		if !strings.Contains(r.URL.RawQuery, "minCudaVersion=12.8") {
			t.Error("CUDA floor missing")
		}
		_, _ = w.Write([]byte(`{"gpus":[{"id":"NVIDIA A40","memory":48,"secure":true,"availability":"HIGH","price":{"secure":0.59}},{"id":"NVIDIA RTX A5000","memory":24,"secure":true,"availability":"LOW","price":{"secure":0.27}}]}`))
	}))
	defer server.Close()
	api := Runpod{URL: server.URL, Token: "private-key"}
	quote, err := api.Quote(context.Background())
	if err != nil || quote.GPUType != "NVIDIA A40" || quote.HourlyRate != .59 {
		t.Fatalf("unexpected quote: %+v %v", quote, err)
	}
}
func TestCreateBoundsHardwareAndDoesNotExposeManagementKey(t *testing.T) {
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		gpu := body["gpu"].(map[string]any)
		if gpu["count"] != float64(1) || gpu["minCudaVersion"] != "12.8" {
			t.Error("unbounded placement")
		}
		env := body["env"].(map[string]any)
		if env["RUNPOD_API_KEY"] != "" {
			t.Error("management key was forwarded")
		}
		if _, ok := body["mounts"]; ok {
			t.Error("unexpected persistent billed storage")
		}
		w.WriteHeader(503)
	}))
	defer server.Close()
	api := Runpod{URL: server.URL, Token: "private-key"}
	_, err := api.Create(context.Background(), cp.Allocation{ID: "allocation", ResourceName: "owned", Image: "image@sha256:abc", WorkerToken: "attempt-token"}, "")
	if err == nil || calls != 1 {
		t.Fatal("provider mutation was retried")
	}
	if strings.Contains(err.Error(), "private-key") {
		t.Fatal("secret in error")
	}
}
func TestMissingTerminationIsIdempotent(t *testing.T) {
	server := httptest.NewServer(http.NotFoundHandler())
	defer server.Close()
	api := Runpod{URL: server.URL}
	if err := api.Delete(context.Background(), "absent"); err != nil {
		t.Fatal(err)
	}
}

func TestTwoGPUsUseSameHostAndReserveFullPrice(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "GET" {
			if r.URL.Query().Get("count") != "2" {
				t.Error("same-host two-GPU capacity not requested")
			}
			_, _ = w.Write([]byte(`{"gpus":[{"id":"NVIDIA A40","memory":48,"secure":true,"availability":"HIGH","price":{"secure":0.59}}]}`))
			return
		}
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		if body["gpu"].(map[string]any)["count"] != float64(2) {
			t.Error("two GPU Pod not requested")
		}
		_, _ = w.Write([]byte(`{"id":"two","gpu":{"count":2},"cost":1.18}`))
	}))
	defer server.Close()
	api := Runpod{URL: server.URL}
	quote, err := api.QuoteFor(context.Background(), 2)
	if err != nil || quote.HourlyRate != 1.18 || quote.GPUCount != 2 {
		t.Fatalf("incorrect two-GPU quote: %+v %v", quote, err)
	}
	pod, err := api.Create(context.Background(), cp.Allocation{GPUCount: 2}, "")
	if err != nil || pod.GPU.Count != 2 {
		t.Fatal("two-GPU Pod contract failed")
	}
}

func TestRegistryCleanupReconcilesAnAmbiguousCreateByOwnedName(t *testing.T) {
	deleted := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "DELETE" {
			if r.URL.Path != "/registries/owned-registry" {
				t.Fatal("foreign registry deletion")
			}
			deleted = true
			return
		}
		if deleted {
			_, _ = w.Write([]byte(`{"registries":[{"id":"foreign","name":"another-project"}]}`))
		} else {
			_, _ = w.Write([]byte(`{"registries":[{"id":"owned-registry","name":"qps-allocation"},{"id":"foreign","name":"another-project"}]}`))
		}
	}))
	defer server.Close()
	api := Runpod{URL: server.URL}
	if api.CleanupRegistries(context.Background(), "qps-allocation", "") == nil || !deleted {
		t.Fatal("cleanup completed before provider confirmation")
	}
	if err := api.CleanupRegistries(context.Background(), "qps-allocation", ""); err != nil {
		t.Fatal(err)
	}
}
