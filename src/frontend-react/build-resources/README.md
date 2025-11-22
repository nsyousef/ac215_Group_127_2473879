# App Icon Setup

## Generating the macOS App Icon

To create the app icon with the proper macOS squircle shape:

1. Start a local HTTP server from the `frontend-react` directory:
   ```bash
   cd src/frontend-react
   python3 -m http.server 8080
   ```

2. Open the icon generator in your browser:
   ```
   http://localhost:8080/scripts/generate-icon.html
   ```

3. The icon will be automatically generated with the pibu logo (blue background with white text)

4. Download all sizes by clicking the buttons:
   - 1024x1024 (for .icns generation)
   - 512x512 (main icon)
   - 256x256
   - 128x128

Note: The HTTP server is required because the generator needs to load the SVG file from `public/assets/pibu_logo.svg` without CORS restrictions.

## Creating .icns for macOS

After downloading the PNG files, use the following commands to create the .icns file:

```bash
# Create iconset directory
mkdir icon.iconset

# Create all required sizes (macOS requires specific sizes)
# You can use ImageMagick or sips (built into macOS)

# Using sips (macOS native):
sips -z 16 16     icon_1024x1024.png --out icon.iconset/icon_16x16.png
sips -z 32 32     icon_1024x1024.png --out icon.iconset/icon_16x16@2x.png
sips -z 32 32     icon_1024x1024.png --out icon.iconset/icon_32x32.png
sips -z 64 64     icon_1024x1024.png --out icon.iconset/icon_32x32@2x.png
sips -z 128 128   icon_1024x1024.png --out icon.iconset/icon_128x128.png
sips -z 256 256   icon_1024x1024.png --out icon.iconset/icon_128x128@2x.png
sips -z 256 256   icon_1024x1024.png --out icon.iconset/icon_256x256.png
sips -z 512 512   icon_1024x1024.png --out icon.iconset/icon_256x256@2x.png
sips -z 512 512   icon_1024x1024.png --out icon.iconset/icon_512x512.png
sips -z 1024 1024 icon_1024x1024.png --out icon.iconset/icon_512x512@2x.png

# Convert to .icns
iconutil -c icns icon.iconset

# Move to build-resources
mv icon.icns build-resources/

# Copy the 512x512 as icon.png for Windows/Linux
cp icon_512x512.png build-resources/icon.png

# Clean up
rm -rf icon.iconset
```

## Quick Setup Script

Run this from the frontend-react directory after generating the icons:

```bash
cd src/frontend-react

# Assuming you've downloaded icon_1024x1024.png to the current directory
mkdir -p icon.iconset

# Generate all sizes
for size in 16 32 128 256 512; do
  sips -z $size $size icon_1024x1024.png --out icon.iconset/icon_${size}x${size}.png
done

# Generate @2x versions
sips -z 32 32 icon_1024x1024.png --out icon.iconset/icon_16x16@2x.png
sips -z 64 64 icon_1024x1024.png --out icon.iconset/icon_32x32@2x.png
sips -z 256 256 icon_1024x1024.png --out icon.iconset/icon_128x128@2x.png
sips -z 512 512 icon_1024x1024.png --out icon.iconset/icon_256x256@2x.png
sips -z 1024 1024 icon_1024x1024.png --out icon.iconset/icon_512x512@2x.png

# Create .icns
iconutil -c icns icon.iconset

# Move files
mv icon.icns build-resources/
cp icon.iconset/icon_512x512.png build-resources/icon.png

# Cleanup
rm -rf icon.iconset icon_*.png
```

## Configuration

The `electron-package.json` has been updated to:
- Use `build-resources` directory for icons
- Point to `icon.icns` for macOS builds
- Point to `icon.png` for Windows/Linux builds
- Set product name to "pibu.ai"
- Categorize app as healthcare-fitness

### Development vs Packaged Builds (macOS)
- In packaged apps (built with `electron-builder`), macOS uses the `.icns` file automatically.
- In development (`npm run dev-electron`), macOS ignores `BrowserWindow.icon` for the Dock. We've added code to set the Dock icon explicitly from `build-resources/icon.png` when the app starts.
- If you still see the Electron icon, ensure `build-resources/icon.png` exists and restart the dev app:
   ```bash
   npm run dev-electron
   ```

## Testing

To test the icon in development, you can add this to `electron/main.js`:

```javascript
const icon = path.join(__dirname, '..', 'build-resources', 'icon.png');
mainWindow = new BrowserWindow({
  icon: icon, // Add this line
  // ... rest of config
});
```
