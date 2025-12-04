/** @type {import('next').NextConfig} */
const nextConfig = {
    // Only use static export for production builds (Electron), not for dev server
    output: process.env.NODE_ENV === 'production' ? 'export' : undefined,
    distDir: 'out',
    reactStrictMode: false,
    images: {
        unoptimized: true  // Required for static export (Electron)
    },
    webpack: (config) => {
        config.module.rules.push({
            test: /\.svg$/,
            use: ["@svgr/webpack"]
        });
        return config;
    },
};

module.exports = nextConfig;
