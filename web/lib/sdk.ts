import createClient from 'openapi-fetch';
import type { paths, components } from './api-types';

export type Submission = components['schemas']['HostedSubmission'];
export const client = createClient<paths>({ credentials: 'same-origin' });
let csrf = '';
export function setSDKCSRF(value: string) { csrf = value; }
client.use({ onRequest({ request }) { if (csrf) request.headers.set('X-CSRF-Token', csrf); return request; } });
