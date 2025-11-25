'use client'

import { useState } from 'react'
import Link from 'next/link';
import Image from 'next/image';
import { usePathname } from 'next/navigation';
import { Home, Menu, X } from 'lucide-react';
import ThemeToggle from './ThemeToggle';

export default function Header() {
    // Component States
    const [isMenuOpen, setIsMenuOpen] = useState(false);
    const pathname = usePathname();

    // Add your navigation items here
    const navItems = [
        { name: 'Home', path: '/', icon: <Home className="h-5 w-5" /> }
        // Add more navigation items as needed
        // Example: { name: 'About', path: '/about', icon: <Info className="h-5 w-5" /> }
    ];

    // UI View
    return (
        <>
            <header className="header-wrapper">
                <div className="header-container">
                    <div className="header-content">
                        <Link href="/" className="header-logo">
                            <Image
                                src="/assets/logo_square.png"
                                alt="App Logo"
                                width={32}
                                height={32}
                                className="w-8 h-8"
                            />
                            <h1 className="text-xl font-bold">Your App Name</h1>
                        </Link>

                        <nav className="nav-desktop">
                            {navItems.map((item) => (
                                <Link
                                    key={item.name}
                                    href={item.path}
                                    className={`nav-link ${pathname === item.path ? 'nav-link-active' : ''}`}
                                >
                                    <div className="nav-icon-wrapper">{item.icon}</div>
                                    <span className="nav-text">{item.name}</span>
                                </Link>
                            ))}
                        </nav>

                        <div className="flex items-center gap-2">
                            <ThemeToggle />
                            <button
                                className="mobile-menu-button"
                                onClick={() => setIsMenuOpen(!isMenuOpen)}
                                aria-label="Toggle menu"
                            >
                                {isMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
                            </button>
                        </div>
                    </div>
                </div>

                {/* Mobile Menu */}
                {isMenuOpen && (
                    <div className="mobile-menu translate-y-0">
                        <nav className="px-4 py-2 space-y-1">
                            {navItems.map((item) => (
                                <Link
                                    key={item.name}
                                    href={item.path}
                                    className={`flex items-center gap-3 px-3 py-2 rounded-md text-muted-foreground hover:text-foreground hover:bg-accent transition-all ${
                                        pathname === item.path ? 'bg-accent text-foreground font-medium' : ''
                                    }`}
                                    onClick={() => setIsMenuOpen(false)}
                                >
                                    {item.icon}
                                    <span>{item.name}</span>
                                </Link>
                            ))}
                        </nav>
                    </div>
                )}
            </header>
            {isMenuOpen && <div className="mobile-menu-overlay" onClick={() => setIsMenuOpen(false)} />}
        </>
    );
}
