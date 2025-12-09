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
  // Front view
  'ear': { leftPct: 40, topPct: 12 },
  'hair': { leftPct: 50, topPct: 6 },
  'face': { leftPct: 50, topPct: 12 },
  'neck': { leftPct: 50, topPct: 17 },
  'chest': { leftPct: 50, topPct: 30 },
  'abdomen': { leftPct: 50, topPct: 42 },
  'shoulders': { leftPct: 50, topPct: 22 },
  'armpit': { leftPct: 41, topPct: 27 },
  'upper arm': { leftPct: 37, topPct: 30 },
  'lower arm': { leftPct: 36, topPct: 42 },
  'hands': { leftPct: 36, topPct: 55 },
  'groin': { leftPct: 50, topPct: 54 },
  'hips': { leftPct: 42, topPct: 54 },
  'thighs': { leftPct: 50, topPct: 64 },
  'lower legs': { leftPct: 50, topPct: 80 },
  'foot': { leftPct: 50, topPct: 93 },
  // Back view
  'back of head': { leftPct: 50, topPct: 6 },
  'upper back': { leftPct: 50, topPct: 22 },
  'mid back': { leftPct: 50, topPct: 30 },
  'lower back': { leftPct: 50, topPct: 42 },
  'buttocks': { leftPct: 50, topPct: 54 },
  'calves': { leftPct: 50, topPct: 80 },
};

export const PAGES = {
    HOME: 'list',
    BODY_MAP: 'bodyMap',
    RESULTS: 'results',
};
