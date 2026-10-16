package provider

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	cp "github.com/Ayobami-00/qwen3-posttrain-and-serve-platform/operator/internal/controlplane"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/ecr"
)

var ErrNotFound = errors.New("provider resource absent")

type HTTPError struct{ Status int }

func (e HTTPError) Error() string { return fmt.Sprintf("provider returned HTTP %d", e.Status) }

type Quote struct {
	HourlyRate float64 `json:"hourlyRate"`
	GPUType    string  `json:"gpuType"`
	GPUCount   int     `json:"gpuCount"`
}
type Pod struct {
	ID     string  `json:"id"`
	Name   string  `json:"name"`
	Status string  `json:"status"`
	Cost   float64 `json:"cost"`
	Image  string  `json:"image"`
	GPU    struct {
		Count int `json:"count"`
	} `json:"gpu"`
}
type Registry struct{ ID, Name string }
type GPU struct {
	ID           string `json:"id"`
	Memory       int    `json:"memory"`
	Secure       bool   `json:"secure"`
	Availability string `json:"availability"`
	Price        struct {
		Secure float64 `json:"secure"`
	} `json:"price"`
}
type Runpod struct {
	URL, Token, Region string
	HTTP               *http.Client
}

func (r *Runpod) request(ctx context.Context, method, path string, body, out any) error {
	var encoded io.Reader
	if body != nil {
		value, err := json.Marshal(body)
		if err != nil {
			return err
		}
		encoded = bytes.NewReader(value)
	}
	req, err := http.NewRequestWithContext(ctx, method, strings.TrimRight(r.URL, "/")+path, encoded)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+r.Token)
	req.Header.Set("Content-Type", "application/json")
	client := r.HTTP
	if client == nil {
		client = &http.Client{Timeout: 25 * time.Second}
	}
	response, err := client.Do(req)
	if err != nil {
		return errors.New("provider transport unavailable; outcome may be uncertain")
	}
	defer response.Body.Close()
	if response.StatusCode == 404 {
		return ErrNotFound
	}
	if response.StatusCode >= 400 {
		return HTTPError{Status: response.StatusCode}
	}
	if out != nil {
		return json.NewDecoder(io.LimitReader(response.Body, 8<<20)).Decode(out)
	}
	return nil
}
func (r *Runpod) Quote(ctx context.Context) (Quote, error) { return r.QuoteFor(ctx, 1) }
func (r *Runpod) QuoteFor(ctx context.Context, count int) (Quote, error) {
	if count != 1 && count != 2 {
		return Quote{}, errors.New("unsupported GPU topology")
	}
	var payload struct {
		GPUs []GPU `json:"gpus"`
	}
	err := r.request(ctx, "GET", fmt.Sprintf("/catalog/gpus?include=AVAILABILITY&product=POD&count=%d&cloud=SECURE&minCudaVersion=12.8", count), nil, &payload)
	if err != nil {
		return Quote{}, err
	}
	// A40 is the recorded compatibility baseline; alternatives still run the worker's CUDA/BF16 probe.
	for _, name := range []string{"NVIDIA A40", "NVIDIA RTX A5000", "NVIDIA L4", "NVIDIA GeForce RTX 4090"} {
		for _, gpu := range payload.GPUs {
			if gpu.ID == name && gpu.Secure && gpu.Memory >= 24 && gpu.Availability != "NONE" && gpu.Availability != "" && gpu.Price.Secure > 0 && gpu.Price.Secure <= 1 {
				return Quote{HourlyRate: gpu.Price.Secure * float64(count), GPUType: gpu.ID, GPUCount: count}, nil
			}
		}
	}
	return Quote{}, errors.New("no compatible GPU within the price ceiling")
}
func (r *Runpod) List(ctx context.Context) ([]Pod, error) {
	var all []Pod
	cursor := ""
	for page := 0; page < 100; page++ {
		var response struct {
			Pods       []Pod `json:"pods"`
			Pagination struct {
				NextCursor string `json:"nextCursor"`
			} `json:"pagination"`
			NextCursor string `json:"nextCursor"`
		}
		path := "/pods?limit=100"
		if cursor != "" {
			path += "&cursor=" + url.QueryEscape(cursor)
		}
		if err := r.request(ctx, "GET", path, nil, &response); err != nil {
			return nil, err
		}
		all = append(all, response.Pods...)
		next := response.Pagination.NextCursor
		if next == "" {
			next = response.NextCursor
		}
		if next == "" {
			return all, nil
		}
		if next == cursor {
			return nil, errors.New("provider pagination repeated a cursor")
		}
		cursor = next
	}
	return nil, errors.New("provider pagination exceeded safe limit")
}
func (r *Runpod) Get(ctx context.Context, id string) (Pod, error) {
	var pod Pod
	err := r.request(ctx, "GET", "/pods/"+url.PathEscape(id), nil, &pod)
	return pod, err
}
func (r *Runpod) Create(ctx context.Context, a cp.Allocation, registry string) (Pod, error) {
	count := a.GPUCount
	if count == 0 {
		count = 1
	} // Legacy single-GPU allocation contracts.
	if count != 1 && count != 2 {
		return Pod{}, errors.New("unsupported GPU topology")
	}
	env := map[string]string{"QPS_API_URL": a.APIURL, "QPS_ALLOCATION_ID": a.ID, "QPS_WORKER_TOKEN": a.WorkerToken, "QPS_IMAGE_DIGEST": a.Image, "QPS_WORK_DIR": "/workspace/qps", "RUNPOD_API_KEY": ""}
	body := map[string]any{"name": a.ResourceName, "image": a.Image, "cloud": "SECURE", "disk": 50,
		"gpu": map[string]any{"id": a.GPUType, "count": count, "minCudaVersion": "12.8", "minRamPerGpu": 16, "minVcpuCountPerGpu": 4},
		"env": env, "entrypoint": []string{"python", "-m", "qps.hosted.agent"}, "cmd": []string{}, "startSsh": false, "startJupyter": false}
	if a.Kind == "deployment" {
		body["ports"] = []string{"8000/http"}
	} else {
		body["ports"] = []string{}
	}
	if registry != "" {
		body["registry"] = registry
	}
	var pod Pod
	err := r.request(ctx, "POST", "/pods", body, &pod)
	return pod, err
}
func (r *Runpod) Delete(ctx context.Context, id string) error {
	err := r.request(ctx, "DELETE", "/pods/"+url.PathEscape(id), nil, nil)
	if errors.Is(err, ErrNotFound) {
		return nil
	}
	return err
}
func (r *Runpod) EnsureRegistry(ctx context.Context, a cp.Allocation) (string, error) {
	if !strings.Contains(a.Image, ".dkr.ecr.") {
		return "", nil
	}
	var response struct {
		Registries []Registry `json:"registries"`
	}
	if err := r.request(ctx, "GET", "/registries", nil, &response); err != nil {
		return "", err
	}
	for _, v := range response.Registries {
		if v.Name == a.ResourceName {
			return v.ID, nil
		}
	}
	configuration, err := config.LoadDefaultConfig(ctx, config.WithRegion(r.Region))
	if err != nil {
		return "", errors.New("ECR role credentials unavailable")
	}
	token, err := ecr.NewFromConfig(configuration).GetAuthorizationToken(ctx, &ecr.GetAuthorizationTokenInput{})
	if err != nil || len(token.AuthorizationData) == 0 {
		return "", errors.New("ECR pull token unavailable")
	}
	decoded, err := base64.StdEncoding.DecodeString(*token.AuthorizationData[0].AuthorizationToken)
	if err != nil {
		return "", errors.New("invalid ECR token")
	}
	pair := strings.SplitN(string(decoded), ":", 2)
	if len(pair) != 2 {
		return "", errors.New("invalid ECR token")
	}
	var created Registry
	err = r.request(ctx, "POST", "/registries", map[string]string{"name": a.ResourceName, "username": pair[0], "password": pair[1]}, &created)
	return created.ID, err
}
func (r *Runpod) DeleteRegistry(ctx context.Context, id string) error {
	if id == "" {
		return nil
	}
	err := r.request(ctx, "DELETE", "/registries/"+url.PathEscape(id), nil, nil)
	if errors.Is(err, ErrNotFound) {
		return nil
	}
	return err
}
