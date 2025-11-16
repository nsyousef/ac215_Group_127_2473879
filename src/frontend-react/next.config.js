/** @type {import('next').NextConfig} */
const nextConfig = {
    output: 'export',  // Required for Electron
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
