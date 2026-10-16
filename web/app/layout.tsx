import type { Metadata } from 'next';
import './globals.css';
export const metadata: Metadata = { title: 'QPS · Experiments', description: 'Reproducible Qwen3 training and serving' };
export default function Layout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
