#!/bin/bash

# Quick Start Guide - SkinCare App UI

echo "SkinCare App - Material Design Setup"
echo "========================================"
echo ""

# Change to frontend directory
cd src/frontend-react

echo "Installing dependencies..."
npm install

echo ""
echo "Setup complete!"
echo ""
echo "To start development:"
echo "   npm run dev"
echo ""
echo "To build for production:"
echo "   npm run build"
echo ""
echo "The app will be available at: http://localhost:3000"
echo ""
echo "Documentation:"
echo "   - UI_UPDATES.md - Detailed implementation guide"
echo "   - MATERIAL_UI_MIGRATION.md - Migration summary"
echo "   - UPDATE_CHECKLIST.md - Verification checklist"
echo "   - BEFORE_AFTER_COMPARISON.md - Architecture changes"
echo ""
echo "Current Pages:"
echo "   - Home (List View): /"
echo "   - Body Map: /body-map"
echo "   - Results: /results"
echo ""
echo "Adding New Pages:"
echo "   1. Create src/app/new-page/page.jsx"
echo "   2. Import MobileLayout from '@/components/layouts/MobileLayout'"
echo "   3. Wrap content and use Material UI components"
echo ""
echo "Happy coding!"
