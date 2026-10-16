package controlplane

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type Client struct {
	URL, Token string
	HTTP       *http.Client
}
type Allocation struct {
	ID                  string   `json:"id"`
	OwnerID             string   `json:"owner_id"`
	Kind                string   `json:"kind"`
	Generation          int      `json:"generation"`
	Profile             string   `json:"profile"`
	State               string   `json:"state"`
	Desired             string   `json:"desired"`
	OwnerDesired        string   `json:"ownerDesired"`
	PodID               string   `json:"pod_id"`
	GPUType             string   `json:"gpu_type"`
	HourlyRate          float64  `json:"hourly_rate"`
	Created             float64  `json:"created"`
	Deadline            float64  `json:"deadline"`
	LeaseUntil          float64  `json:"lease_until"`
	Heartbeat           *float64 `json:"heartbeat"`
	Phase               string   `json:"phase"`
	Ready               bool     `json:"ready"`
	Result              *Result  `json:"result"`
	Endpoint            string   `json:"endpoint"`
	WorkerToken         string   `json:"workerToken"`
	APIURL              string   `json:"apiUrl"`
	Image               string   `json:"image"`
	ResourceName        string   `json:"resourceName"`
	Namespace           string   `json:"namespace"`
	ExpectedArtifactID  string   `json:"expectedArtifactId"`
	ExpectedManifestSHA string   `json:"expectedManifestSha"`
}
type Result struct{ Status, Reason string }
type Workload struct {
	ID         string `json:"id"`
	Status     string `json:"status"`
	Desired    string `json:"desired"`
	Attempt    int    `json:"attempt"`
	Generation int    `json:"generation"`
	Submission struct {
		ComputeProfile string `json:"computeProfile"`
	} `json:"submission"`
	Spec struct {
		ComputeProfile string `json:"computeProfile"`
	} `json:"spec"`
	Allocations []Allocation `json:"allocations"`
}

func (w Workload) Revision(kind string) int {
	if kind == "training" {
		return w.Attempt
	}
	return w.Generation
}
func (w Workload) Profile(kind string) string {
	if kind == "training" {
		return w.Submission.ComputeProfile
	}
	return w.Spec.ComputeProfile
}

type HTTPError struct{ Status int }

func (e HTTPError) Error() string { return fmt.Sprintf("control plane returned HTTP %d", e.Status) }
func (c *Client) Request(ctx context.Context, method, path string, body, out any) error {
	var payload io.Reader
	if body != nil {
		encoded, err := json.Marshal(body)
		if err != nil {
			return err
		}
		payload = bytes.NewReader(encoded)
	}
	req, err := http.NewRequestWithContext(ctx, method, strings.TrimRight(c.URL, "/")+"/api/internal/v1"+path, payload)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+c.Token)
	req.Header.Set("Content-Type", "application/json")
	client := c.HTTP
	if client == nil {
		client = &http.Client{Timeout: 30 * time.Second}
	}
	response, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("control plane transport unavailable")
	}
	defer response.Body.Close()
	if response.StatusCode >= 400 {
		return HTTPError{response.StatusCode}
	}
	if out != nil {
		return json.NewDecoder(io.LimitReader(response.Body, 4<<20)).Decode(out)
	}
	return nil
}
func (c *Client) Workload(ctx context.Context, kind, id string) (Workload, error) {
	var v Workload
	err := c.Request(ctx, "GET", "/workloads/"+kind+"/"+id, nil, &v)
	return v, err
}
func (c *Client) Allocation(ctx context.Context, id string) (Allocation, error) {
	var v Allocation
	err := c.Request(ctx, "GET", "/allocations/"+id, nil, &v)
	return v, err
}
func (c *Client) Observe(ctx context.Context, id string, body any) (Allocation, error) {
	var v Allocation
	err := c.Request(ctx, "POST", "/allocations/"+id+"/observe", body, &v)
	return v, err
}
func (c *Client) Reserve(ctx context.Context, kind, id string, quote any) (Allocation, error) {
	var v Allocation
	err := c.Request(ctx, "POST", "/workloads/"+kind+"/"+id+"/reserve", quote, &v)
	return v, err
}
func (c *Client) Project(ctx context.Context, kind, id, status, phase string, generation int) error {
	return c.Request(ctx, "POST", "/workloads/"+kind+"/"+id+"/project", map[string]any{"status": status, "phase": phase, "generation": generation}, nil)
}
