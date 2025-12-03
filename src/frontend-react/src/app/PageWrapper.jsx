'use client';

import { useSearchParams } from 'next/navigation';
import { useState, useEffect } from 'react';
import PageContent from './PageContent';

export default function PageWrapper() {
  const searchParams = useSearchParams();
  const viewParam = searchParams.get('view') || 'home';

  // State to track if we're mounted (to avoid hydration mismatch)
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Return loading state until mounted (avoids hydration mismatch)
  if (!mounted) {
    return null;
  }

  return <PageContent initialView={viewParam} />;
}
