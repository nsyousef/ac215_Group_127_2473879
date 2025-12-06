import { Suspense } from 'react';
import InitializingScreen from '../components/InitializingScreen';
import PageWrapper from './PageWrapper';

function LoadingFallback() {
    return <div>Loading...</div>;
}

export default function Home() {
    return (
        <>
            <InitializingScreen />
            <Suspense fallback={<LoadingFallback />}>
                <PageWrapper />
            </Suspense>
        </>
    );
}
