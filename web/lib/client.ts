import { setSDKCSRF } from './sdk';
let csrf = '';
export function setCSRF(value: string) { csrf = value; setSDKCSRF(value); }
export async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(path, { ...options, credentials: 'same-origin', cache: 'no-store',
    headers: { 'Content-Type': 'application/json', ...(csrf ? { 'X-CSRF-Token': csrf } : {}), ...options.headers } });
  const value = await response.json();
  if (!response.ok) throw new Error(`${typeof value.detail === 'string' ? value.detail : `Request failed (${response.status})`} · ${path}`);
  return value as T;
}
export const post = <T>(path: string, body?: unknown) => api<T>(path, { method: 'POST', ...(body === undefined ? {} : { body: JSON.stringify(body) }) });
export interface Training { method: 'sft'; strategy: 'lora'; dtype: 'bf16' | 'fp32'; maxSteps: number; microBatchSize: number; gradientAccumulationSteps: number; learningRate: number; maxSequenceLength: number; checkpointEverySteps: number; seed: number; lora: { rank: number; alpha: number; dropout: number; targetModules: string[] } }
export interface Artifact { id: string; run_id: string; kind: string; manifest_sha: string; size: number; created: number; manifest?: { files: Record<string, { size: number; sha256: string }>; metadata: Record<string, unknown> } }
export interface Attempt { number: number; status: string; checkpoint: string | null; reason: string | null; outputs: Record<string, unknown> }
export interface Allocation { id: string; state: string; desired: string; phase: string; pod_id: string | null; gpu_type: string; hourly_rate: number; heartbeat: number | null; deadline: number }
export interface Run { id: string; name: string; status: string; phase: string; attempt: number; created: number; model: string; dataset: string; training: Training; submission?: { run: Record<string, unknown>; computeProfile: string }; resolved?: Record<string, unknown>; artifacts?: Artifact[]; attempts?: Attempt[]; allocations?: Allocation[] }
export interface Metric { id: number; attempt: number; time: string; kind: string; payload: Record<string, number | null> }
export interface GPU { index: number; name: string; utilization_percent: number | null; memory_used_bytes: number | null; memory_total_bytes: number | null; temperature_celsius: number | null; power_watts: number | null }
export interface Telemetry { attempt: number; workerGroup: string; worker: string; phase: string; stale: boolean; lastSeen: number | null; gpus: GPU[] }
export interface Deployment { id: string; name: string; status: string; spec: { artifactId: string; computeProfile: string } }
export interface Event { id: number; kind: string; attempt: number; time: string; payload: Record<string, unknown> }
