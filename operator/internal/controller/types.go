package controller

import (
	"encoding/json"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

var GroupVersion = schema.GroupVersion{Group: "qps.dev", Version: "v1alpha1"}

type WorkloadSpec struct {
	OwnerID    string `json:"ownerId"`
	Generation int    `json:"generation"`
	Action     string `json:"action"`
}
type WorkloadStatus struct {
	Phase              string `json:"phase,omitempty"`
	AllocationID       string `json:"allocationId,omitempty"`
	ObservedGeneration int64  `json:"observedGeneration,omitempty"`
	Message            string `json:"message,omitempty"`
}
type TrainingRun struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`
	Spec              WorkloadSpec   `json:"spec"`
	Status            WorkloadStatus `json:"status,omitempty"`
}
type TrainingRunList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []TrainingRun `json:"items"`
}
type ModelDeployment struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`
	Spec              WorkloadSpec   `json:"spec"`
	Status            WorkloadStatus `json:"status,omitempty"`
}
type ModelDeploymentList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []ModelDeployment `json:"items"`
}
type AllocationSpec struct {
	AllocationID string `json:"allocationId"`
	OwnerID      string `json:"ownerId"`
	OwnerKind    string `json:"ownerKind"`
	Generation   int    `json:"generation"`
}
type AllocationStatus struct {
	Phase           string `json:"phase,omitempty"`
	PodID           string `json:"podId,omitempty"`
	RegistryID      string `json:"registryId,omitempty"`
	CreateRequested bool   `json:"createRequested,omitempty"`
	RequestedAt     int64  `json:"requestedAt,omitempty"`
	Message         string `json:"message,omitempty"`
}
type ComputeAllocation struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`
	Spec              AllocationSpec   `json:"spec"`
	Status            AllocationStatus `json:"status,omitempty"`
}
type ComputeAllocationList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []ComputeAllocation `json:"items"`
}

func clone[T any](in *T) *T {
	value, _ := json.Marshal(in)
	out := new(T)
	_ = json.Unmarshal(value, out)
	return out
}
func (x *TrainingRun) DeepCopyObject() runtime.Object           { return clone(x) }
func (x *TrainingRunList) DeepCopyObject() runtime.Object       { return clone(x) }
func (x *ModelDeployment) DeepCopyObject() runtime.Object       { return clone(x) }
func (x *ModelDeploymentList) DeepCopyObject() runtime.Object   { return clone(x) }
func (x *ComputeAllocation) DeepCopyObject() runtime.Object     { return clone(x) }
func (x *ComputeAllocationList) DeepCopyObject() runtime.Object { return clone(x) }
func AddToScheme(s *runtime.Scheme) error {
	s.AddKnownTypes(GroupVersion, &TrainingRun{}, &TrainingRunList{}, &ModelDeployment{}, &ModelDeploymentList{}, &ComputeAllocation{}, &ComputeAllocationList{})
	metav1.AddToGroupVersion(s, GroupVersion)
	return nil
}
