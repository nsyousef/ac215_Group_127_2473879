'use client';

import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';

export default function Home() {
    return (
        <div className="min-h-screen bg-background">
            {/* Hero Section */}
            <section className="relative py-20 px-4 sm:px-6 lg:px-8">
                <div className="max-w-4xl mx-auto text-center">
                    <h1 className="text-4xl sm:text-5xl font-bold tracking-tight mb-6">
                        Welcome to Your App
                    </h1>
                    <p className="text-lg text-muted-foreground max-w-2xl mx-auto mb-8">
                        This is a minimal starter template. Customize this page to fit your application needs.
                    </p>
                    <div className="flex flex-wrap justify-center gap-4">
                        <Button size="lg">
                            Get Started
                        </Button>
                        <Button size="lg" variant="outline">
                            Learn More
                        </Button>
                    </div>
                </div>
            </section>

            {/* Content Section */}
            <section className="py-16 px-4 sm:px-6 lg:px-8">
                <div className="max-w-4xl mx-auto">
                    <Card className="p-8">
                        <h2 className="text-2xl font-semibold mb-4">Getting Started</h2>
                        <p className="text-muted-foreground mb-4">
                            This template includes:
                        </p>
                        <ul className="list-disc list-inside space-y-2 text-muted-foreground">
                            <li>Next.js 15 with React 18</li>
                            <li>Tailwind CSS for styling</li>
                            <li>Dark mode support with theme toggle</li>
                            <li>Basic UI components from shadcn/ui</li>
                            <li>Responsive layout with header and footer</li>
                        </ul>
                        <p className="text-muted-foreground mt-4">
                            Start building by editing the files in the <code className="bg-muted px-2 py-1 rounded">src/app</code> directory.
                        </p>
                    </Card>
                </div>
            </section>
        </div>
    );
}
