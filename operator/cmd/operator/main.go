package main

import (
	"context"
	"encoding/json"
	"github.com/Ayobami-00/qwen3-posttrain-and-serve-platform/operator/internal/controller"
	"github.com/Ayobami-00/qwen3-posttrain-and-serve-platform/operator/internal/controlplane"
	"github.com/Ayobami-00/qwen3-posttrain-and-serve-platform/operator/internal/provider"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/secretsmanager"
	"k8s.io/apimachinery/pkg/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"log"
	"os"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
)

func env(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}
func main() {
	ctrl.SetLogger(zap.New(zap.UseDevMode(false)))
	ctx := context.Background()
	values := map[string]string{}
	if arn := os.Getenv("QPS_OPERATOR_SECRET_ARN"); arn != "" {
		cfg, err := config.LoadDefaultConfig(ctx)
		if err != nil {
			log.Fatal("AWS role credentials unavailable")
		}
		result, err := secretsmanager.NewFromConfig(cfg).GetSecretValue(ctx, &secretsmanager.GetSecretValueInput{SecretId: &arn})
		if err != nil || result.SecretString == nil {
			log.Fatal("Operator secret unavailable")
		}
		if err := json.Unmarshal([]byte(*result.SecretString), &values); err != nil {
			log.Fatal("Operator secret format invalid")
		}
	}
	token := env("QPS_INTERNAL_TOKEN", values["internal_token"])
	providerToken := env("RUNPOD_API_KEY", values["runpod_api_key"])
	if len(token) < 32 {
		log.Fatal("Controller internal token is required")
	}
	namespace := env("QPS_NAMESPACE", "qps")
	scheme := runtime.NewScheme()
	_ = clientgoscheme.AddToScheme(scheme)
	_ = controller.AddToScheme(scheme)
	manager, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{Scheme: scheme,
		LeaderElection: true, LeaderElectionID: "qps-controller.qps.dev", LeaderElectionNamespace: namespace,
		Metrics: metricsserver.Options{BindAddress: ":8081"}, HealthProbeBindAddress: ":8082",
		Cache: cache.Options{DefaultNamespaces: map[string]cache.Config{namespace: {}}}})
	if err != nil {
		log.Fatal("Kubernetes manager setup failed")
	}
	api := &controlplane.Client{URL: env("QPS_API_URL", "http://qps-api:8080"), Token: token}
	cloud := &provider.Runpod{URL: env("RUNPOD_API_URL", "https://api.runpod.io/v2"), Token: providerToken, Region: env("AWS_REGION", "eu-west-2")}
	for _, kind := range []string{"training", "deployment"} {
		r := &controller.WorkloadReconciler{Client: manager.GetClient(), API: api, Provider: cloud, Kind: kind}
		if err := r.SetupWithManager(manager); err != nil {
			log.Fatal("Workload controller setup failed")
		}
	}
	allocation := &controller.AllocationReconciler{Client: manager.GetClient(), API: api, Provider: cloud}
	if err := allocation.SetupWithManager(manager); err != nil {
		log.Fatal("Allocation controller setup failed")
	}
	_ = manager.AddHealthzCheck("healthz", healthz.Ping)
	_ = manager.AddReadyzCheck("readyz", healthz.Ping)
	if err := manager.Start(ctrl.SetupSignalHandler()); err != nil {
		log.Fatal("Controller stopped unexpectedly")
	}
}
