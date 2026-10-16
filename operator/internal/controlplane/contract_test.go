package controlplane

import (
	"encoding/json"
	"os"
	"testing"
)

func TestAllocationSharedWithPython(t *testing.T) {
	data, err := os.ReadFile("../../../contracts/fixtures/allocation.json")
	if err != nil {
		t.Fatal(err)
	}
	var value Allocation
	if err := json.Unmarshal(data, &value); err != nil {
		t.Fatal(err)
	}
	if value.ID != "alloc-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" || value.OwnerID != "run-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" || value.Generation != 1 || value.Profile != "cpu-fixture" || value.Deadline-value.Created != 2700 || value.LeaseUntil-value.Created != 600 || value.WorkerToken != "fixture-only-not-a-valid-token" || value.ResourceName != "qps-test-"+value.ID || value.Namespace != "qps" || value.OwnerDesired != "run" {
		t.Fatalf("Python allocation contract drifted: %+v", value)
	}
	if value.Heartbeat != nil || value.Result != nil || value.Ready {
		t.Fatal("nullable startup state changed")
	}
}
