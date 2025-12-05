// Shared constants for the app

export const CONDITIONS = [
    { id: 1, name: '<disease title>', description: 'Brief Description' },
    { id: 2, name: 'Eczema', description: 'Lorem ipsum dolor sit amet' },
    { id: 3, name: 'Psoriasis', description: 'Lorem ipsum dolor sit amet' },
    { id: 4, name: 'Acne', description: 'Lorem ipsum dolor sit amet' },
    { id: 5, name: 'Mole', description: 'Lorem ipsum dolor sit amet' },
    { id: 6, name: 'Squamous-cell Carcinoma', description: 'Lorem ipsum dolor sit amet' },
    { id: 7, name: 'Leprosy', description: 'Lorem ipsum dolor sit amet' },
];

// Default coordinates for each body part (center of bounding box)
export const BODY_PART_DEFAULTS = {
  'ear': { leftPct: 40, topPct: 12 },
  'hair': { leftPct: 50, topPct: 5 },
  'face': { leftPct: 50, topPct: 12 },
  'neck': { leftPct: 50, topPct: 18 },
  'chest': { leftPct: 50, topPct: 28 },
  'abdomen': { leftPct: 50, topPct: 42 },
  'shoulders': { leftPct: 50, topPct: 22 },
  'upper arm': { leftPct: 50, topPct: 32 },
  'lower arm': { leftPct: 50, topPct: 48 },
  'hands': { leftPct: 50, topPct: 56 },
  'groin': { leftPct: 50, topPct: 58 },
  'thighs': { leftPct: 50, topPct: 68 },
  'lower legs': { leftPct: 50, topPct: 82 },
  'foot': { leftPct: 50, topPct: 94 },
};

export const PAGES = {
    HOME: 'list',
    BODY_MAP: 'bodyMap',
    RESULTS: 'results',
};
