import './globals.css';
import { Providers } from './providers';

export const metadata = {
    title: 'pibu.ai',
    description: 'Dermatological condition detection and tracking',
}

export default function RootLayout({ children }) {
    return (
        <html lang="en" suppressHydrationWarning>
            <head>
                <link href="assets/logo.svg" rel="shortcut icon" type="image/x-icon"></link>
                <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Roboto:300,400,500,700&display=swap" />
                <link rel="stylesheet" href="https://fonts.googleapis.com/icon?family=Material+Icons" />
            </head>
            <body>
                <Providers>
                    {children}
                </Providers>
            </body>
        </html>
    );
}
