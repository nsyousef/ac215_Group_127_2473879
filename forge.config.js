const path = require('path');

module.exports = {
  packagerConfig: {
    name: 'Derma Assistant',
    executableName: 'derma-assistant',
    appBundleId: 'com.dermaassistant.app',
    productName: 'Derma Assistant',
    icon: path.join(__dirname, 'src/frontend-react/build-resources/icon'),
    osxSign: {
      identity: null, // Set to your signing identity if you have one
      'hardened-runtime': true,
      'gatekeeper-assess': false,
      entitlements: path.join(__dirname, 'src/frontend-react/build-resources/entitlements.plist'),
      'entitlements-inherit': path.join(__dirname, 'src/frontend-react/build-resources/entitlements.plist'),
    },
    extraResource: [
      path.join(__dirname, 'src/frontend-react/resources/python-bundle'),
    ],
    files: [
      'out/main',
      'out/renderer',
      'package.json',
      'node_modules',
      'src/frontend-react/electron',
      'src/frontend-react/python',
    ],
  },
  makers: [
    {
      name: '@electron-forge/maker-dmg',
      config: {
        background: path.join(__dirname, 'src/frontend-react/build-resources/dmg-background.png'),
        iconSize: 80,
        format: 'ULFO',
        contents: [
          { x: 220, y: 150, type: 'file', path: 'Derma Assistant.app' },
          { x: 470, y: 150, type: 'link', path: '/Applications' },
        ],
      },
    },
    {
      name: '@electron-forge/maker-zip',
    },
  ],
  publishers: [],
  plugins: [],
};
