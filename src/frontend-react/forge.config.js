const path = require('path');

module.exports = {
  packagerConfig: {
    name: 'pibu_ai',
    productName: 'pibu.ai',
    icon: path.join(__dirname, 'build-resources/icon.icns'),
    // Explicitly set the main entry point
    main: path.join(__dirname, 'electron/main.js'),
    // Build for both x64 and arm64 (M-series) architectures
    arch: ['x64', 'arm64'],
    // Disable asar so postPackage hook can copy static out/ into app folder
    asar: false,
    // Files to include - be very explicit
    files: [
      'electron/',
      'out/',
      'resources/',
      'build-resources/',
      'package.json',
      'node_modules/',
      'python/',
    ],
    // Ignore files using electron-builder patterns (negation patterns in files array)
    ignore: [
      // Development files at root
      '^/\\.next',
      '^/\\.git',
      '^/\\.env',
      '^/\\.DS_Store',
      '^/src$',
      '^/components\\.json$',
      '^/jsconfig\\.json$',
      '^/next\\.config\\.js$',
      '^/tailwind\\.config\\.js$',
      '^/postcss\\.config\\.js$',
      '^/forge\\.config\\.js$',
      '^/Dockerfile',
      '^/docker-shell\\.sh$',
      '^/README\\.md$',
      '^/BUILD_MACOS\\.md$',
      '^/QUICKSTART\\.sh$',
      '^/mockups',
      '^/scripts',
      '^/public',
      '^/dist',
      // Python development artifacts (keep runtime modules)
      '^/python/tests($|/)',
      '^/python/test_.*',
      '^/python/run_.*',
      '^/python/.*\\.ipynb$',
      '^/python/\\.venv',
      '^/python/__pycache__',
      '^/python/inference_local/tests($|/)',
      '^/python/inference_local/.*\\.ipynb$',
      '^/python/Dockerfile',
      '^/python/docker-shell\\.sh$',
      '^/python/\\.env',
      // node_modules cleanup
      '/node_modules/.bin',
      '/node_modules/*/test',
      '/node_modules/*/tests',
      '/node_modules/*/example',
      '/node_modules/*/examples',
      '/node_modules/*/docs',
      '/node_modules/*/README',
      '/node_modules/*/CHANGELOG',
      '/node_modules/*/LICENSE',
    ],
  },
  makers: [
    {
      name: '@electron-forge/maker-dmg',
      config: {
        format: 'ULFO',
        contents: [
          { x: 220, y: 150, type: 'file', path: path.join(__dirname, 'out/pibu_ai-darwin-x64/pibu_ai.app') },
          { x: 470, y: 150, type: 'link', path: '/Applications' },
        ],
      },
    },
    {
      name: '@electron-forge/maker-dmg',
      config: {
        format: 'ULFO',
        contents: [
          { x: 220, y: 150, type: 'file', path: path.join(__dirname, 'out/pibu_ai-darwin-arm64/pibu_ai.app') },
          { x: 470, y: 150, type: 'link', path: '/Applications' },
        ],
      },
    },
  ],
};
