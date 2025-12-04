import { Suspense } from 'react';
import PageWrapper from './PageWrapper';

function LoadingFallback() {
    return <div>Loading...</div>;
}

export default function Home() {
    return (
        <Suspense fallback={<LoadingFallback />}>
            <PageWrapper />
        </Suspense>
    );
}
