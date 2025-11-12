'use client';

import { useTheme } from 'next-themes';
import { useEffect, useState } from 'react';
import { Moon, Sun } from 'lucide-react';
import { Button } from '@/components/ui/button';

export default function ThemeToggle() {
    const { theme, setTheme, resolvedTheme } = useTheme();
    const [mounted, setMounted] = useState(false);

    // Avoid hydration mismatch
    useEffect(() => {
        setMounted(true);
    }, []);

    if (!mounted) {
        return (
            <Button variant="ghost" size="icon" className="w-9 h-9">
                <Sun className="h-4 w-4" />
            </Button>
        );
    }

    // Use resolvedTheme to get the actual current theme (light or dark)
    // even when theme is set to 'system'
    const currentTheme = resolvedTheme || theme;

    return (
        <Button
            variant="ghost"
            size="icon"
            onClick={() => setTheme(currentTheme === 'dark' ? 'light' : 'dark')}
            className="w-9 h-9 hover:bg-accent"
        >
            {currentTheme === 'dark' ? (
                <Sun className="h-4 w-4 text-foreground" />
            ) : (
                <Moon className="h-4 w-4 text-foreground" />
            )}
            <span className="sr-only">Toggle theme</span>
        </Button>
    );
}
