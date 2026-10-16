import type { NextConfig } from 'next';
const api = process.env.QPS_API_INTERNAL_URL || 'http://127.0.0.1:8080';
const config: NextConfig = {
  output: 'standalone',
  poweredByHeader: false,
  async rewrites() { return [
    { source: '/api/:path*', destination: `${api}/api/:path*` },
    { source: '/v1/:path*', destination: `${api}/v1/:path*` },
    { source: '/observability/:path*', destination: `${api}/observability/:path*` },
  ]; },
};
export default config;
