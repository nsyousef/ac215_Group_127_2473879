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
  'head': { leftPct: 50, topPct: 10 },
  'torso': { leftPct: 50, topPct: 36 },
  'left upper arm': { leftPct: 35, topPct: 28 },
  'left lower arm': { leftPct: 35, topPct: 46 },
  'right upper arm': { leftPct: 65, topPct: 28 },
  'right lower arm': { leftPct: 65, topPct: 46 },
  'left hand': { leftPct: 35, topPct: 57 },
  'right hand': { leftPct: 65, topPct: 57 },
  'left upper leg': { leftPct: 45, topPct: 61 },
  'right upper leg': { leftPct: 55, topPct: 61 },
  'left lower leg': { leftPct: 45, topPct: 78 },
  'right lower leg': { leftPct: 55, topPct: 78 },
  'left foot': { leftPct: 45, topPct: 92 },
  'right foot': { leftPct: 55, topPct: 92 },
};

export const PAGES = {
    HOME: 'list',
    BODY_MAP: 'bodyMap',
    RESULTS: 'results',
};
