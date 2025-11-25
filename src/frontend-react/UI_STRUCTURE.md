# 🎨 UI Structure & Component Breakdown

## Page Layouts

### Home Page (`/`)

```
┌─────────────────────────────────────┐
│          [Home]         [+]         │  ← AppBar
├─────────────────────────────────────┤
│                                     │
│  Select a condition to view details │  ← Typography
│                                     │
├─────────────────────────────────────┤
│  ┌──────────────────────────────┐   │
│  │ <disease title>  [🔵] [🔵]   │   │  ← Card (clickable)
│  │ Brief Description             │   │
│  └──────────────────────────────┘   │
│                                     │
│  ┌──────────────────────────────┐   │
│  │ Eczema          [🔵] [🔵]    │   │  ← Card
│  │ Lorem ipsum dolor sit amet   │   │
│  └──────────────────────────────┘   │
│                                     │
│  ┌──────────────────────────────┐   │
│  │ Psoriasis       [🔵] [🔵]    │   │  ← Card
│  │ Lorem ipsum dolor sit amet   │   │
│  └──────────────────────────────┘   │
│                                     │
│  ... (more conditions)              │
│                                     │
│                                     │
│                                     │
├─────────────────────────────────────┤
│  [🏠 List]    [🗺️ Body Map]        │  ← BottomNavigation
└─────────────────────────────────────┘
```

### Body Map Page (`/body-map`)

```
┌─────────────────────────────────────┐
│          [Home]         [+]         │  ← AppBar
├─────────────────────────────────────┤
│                                     │
│  Select a condition to view details │  ← Typography
│                                     │
├─────────────────────────────────────┤
│                                     │
│           ╔════════╗                │
│           ║   ●    ║                │
│           ║  ●●●●  ║                │  ← Body visualization
│           ║  ●   ●  ║                │    with red spots
│           ║   ●●●   ║                │
│           ╚════════╝                │
│                                     │
├─────────────────────────────────────┤
│  ┌──────────────────────────────┐   │
│  │ Tap on the body map         │   │  ← Info Card
│  │ Red dots indicate detected  │   │
│  │ skin conditions             │   │
│  └──────────────────────────────┘   │
│                                     │
│                                     │
├─────────────────────────────────────┤
│  [🏠 List]    [🗺️ Body Map]        │  ← BottomNavigation
└─────────────────────────────────────┘
```

### Results Page (`/results`)

```
┌─────────────────────────────────────┐
│          Results                    │  ← AppBar
├─────────────────────────────────────┤
│                                     │
│  ┌──────────────────────────────┐   │
│  │                              │   │
│  │                              │   │
│  │       [Image Area]           │   │  ← CardMedia / Image
│  │    (nasty_skin.jpg)          │   │
│  │                              │   │
│  │                              │   │
│  └──────────────────────────────┘   │
│                                     │
│  ┌──────────────────────────────┐   │
│  │ Recommendation:              │   │
│  │                              │   │  ← Card with text
│  │ Lorem ipsum dolor sit amet,  │   │
│  │ consectetur adipiscing elit, │   │
│  │ sed do eiusmod tempor...     │   │
│  │                              │   │
│  │ Duis aute irure dolor in     │   │
│  │ reprehenderit in voluptate   │   │
│  │ velit esse cillum dolore     │   │
│  └──────────────────────────────┘   │
│                                     │
│  ┌─────────────────┬───────────────┐│
│  │ 📤 Track Prog.. │ ❓ Ask Quest..││  ← ButtonGroup
│  └─────────────────┴───────────────┘│
│                                     │
└─────────────────────────────────────┘
```

## Component Hierarchy

### MobileLayout Component
```
MobileLayout
├── AppBar
│   ├── Toolbar
│   │   ├── Typography (title)
│   │   └── IconButton (add button)
│   │       └── AddCircleOutline icon
│   │
├── Box (flex container for main content)
│   │
│   └── children (page content)
│       └── Container
│           └── (page-specific content)
│
└── BottomNavigation
    ├── BottomNavigationAction (List)
    │   └── HomeIcon
    └── BottomNavigationAction (Body Map)
        └── MapOutlined icon
```

## Home Page Components

```
Page
├── MobileLayout
│   └── Container
│       ├── Typography (info message)
│       │
│       └── List
│           └── Card (repeating for each condition)
│               └── CardContent
│                   ├── Box (text section)
│                   │   ├── Typography (condition name)
│                   │   └── Typography (description)
│                   │
│                   └── Box (avatar section)
│                       ├── Avatar
│                       └── Avatar
```

## Material Design Components Used

### Structural
- `Box` - Universal layout component
- `Container` - Fixed max-width wrapper
- `List` - List container
- `Card` - Content card
- `CardContent` - Card content wrapper

### Header/Navigation
- `AppBar` - Top navigation bar
- `Toolbar` - Toolbar container
- `BottomNavigation` - Bottom tab navigation
- `BottomNavigationAction` - Navigation item

### Content
- `Typography` - Text with predefined styles
- `Avatar` - Circular image/placeholder
- `Button` - Interactive button
- `ButtonGroup` - Grouped buttons
- `CardMedia` - Card image

### Icons
- `Home as HomeIcon` - Home icon
- `MapOutlined` - Map icon
- `AddCircleOutline` - Add icon
- `AssignmentReturnOutlined` - Track icon
- `QuestionAnswerOutlined` - Question icon

## Styling Pattern

### sx Prop Usage
```jsx
<Box
    sx={{
        // Layout
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 2,

        // Spacing
        p: 2,  // padding
        mb: 1.5,  // margin-bottom

        // Colors
        bgcolor: '#f5f5f5',
        color: '#000',

        // Borders & Shadows
        border: '1px solid #e0e0e0',
        borderRadius: 1,
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',

        // Interactive
        cursor: 'pointer',
        transition: 'all 0.2s',

        // Responsive
        fontSize: { xs: '12px', sm: '14px', md: '16px' },

        // Pseudo-classes
        '&:hover': {
            bgcolor: '#e3f2fd'
        },
        '&.Mui-selected': {
            color: '#1976d2'
        }
    }}
>
    Content
</Box>
```

## State Management

### Home Page
```javascript
const [selectedCondition, setSelectedCondition] = useState(null);

// Used to:
// - Highlight selected card
// - Control card styling (bgcolor, border)
// - Show/hide detailed information
```

### MobileLayout
```javascript
const [value, setValue] = useState(currentPage === 'bodyMap' ? 1 : 0);

// Used to:
// - Highlight active tab
// - Handle page navigation
// - Update router path
```

## Color Palette

```
Primary Colors:
- Primary Blue: #1976d2 (main actions, links, active states)
- Secondary Pink: #dc004e (alternate actions)

Backgrounds:
- Page BG: #f5f5f5 (light gray)
- Card BG: #fff (white)
- Selected Card: #e3f2fd (light blue)

Text:
- Primary Text: #000 (dark)
- Secondary Text: #666 (medium gray)
- Muted Text: #999 (light gray)

Borders:
- Border Color: #e0e0e0 (light gray)
- Avatar BG: #e0e0e0 (light gray)

Status:
- Error/Alert: #e74c3c (red)
```

## Spacing Scale (Material Design)

```
1 = 4px    (xs)
2 = 8px    (sm)
3 = 12px   (md)
4 = 16px   (lg)
6 = 24px   (xl)
8 = 32px   (2xl)
```

**Common Usage**:
- `p: 2` = 8px padding
- `mb: 1.5` = 6px margin-bottom
- `gap: 2` = 8px gap between items

## Responsive Breakpoints

```
xs: 0px      (mobile phones)
sm: 600px    (tablet)
md: 900px    (small laptop)
lg: 1200px   (desktop)
xl: 1536px   (large desktop)
```

**Container**: maxWidth="sm" = max 600px width (optimized for mobile)

---

This structure ensures consistency, scalability, and ease of maintenance across all pages!
