const path = require('path');

module.exports = {
  packagerConfig: {
    name: 'pibu_ai',
    productName: 'pibu.ai',
    icon: path.join(__dirname, 'build-resources/icon.icns'),
    // Explicitly set the main entry point
    main: path.join(__dirname, 'electron/main.js'),
    // Files to include in the packaged app
    files: [
      'electron/',
      'out/',
      '.next/',
      'public/',
      'resources/',
      'build-resources/',
      'package.json',
      'node_modules/',
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
  ],
};
