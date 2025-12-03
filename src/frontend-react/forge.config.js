const path = require('path');

module.exports = {
  packagerConfig: {
    name: 'Derma Assistant',
    productName: 'Derma Assistant',
    icon: path.join(__dirname, 'build-resources/icon.icns'),
  },
  makers: [
    {
      name: '@electron-forge/maker-dmg',
      config: {
        format: 'ULFO',
        contents: [
          { x: 220, y: 150, type: 'file', path: path.join(__dirname, 'out/Derma Assistant-darwin-x64/Derma Assistant.app') },
          { x: 470, y: 150, type: 'link', path: '/Applications' },
        ],
      },
    },
  ],
};
