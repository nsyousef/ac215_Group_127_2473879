'use client';

import { useRef, useState, useEffect } from 'react';
import { Box, Button } from '@mui/material';
import FlipIcon from '@mui/icons-material/Flip';

/**
 * Determines body part label for BACK view based on coordinates
 * @param {number} horizontal - Horizontal coordinate as percentage (0-100)
 * @param {number} vertical - Vertical coordinate as percentage (0-100)
 * @returns {string} Body part label or "invalid"
 */
function getBackBodyPartFromCoordinates(horizontal, vertical) {
  // Overall bounds check
  if (vertical < 4 || vertical > 96 || horizontal < 32 || horizontal > 68) {
    return 'invalid';
  }

  // Back of head: 4-15% vertical
  if (vertical >= 4 && vertical < 15) {
    if (vertical >= 4 && vertical < 8 && horizontal >= 45 && horizontal <= 55) {
      return 'back of head';
    }
    if (vertical >= 8 && horizontal >= 45 && horizontal <= 55) {
      return 'back of head';
    }
    return 'invalid';
  }

  // Neck (back): 15-20% vertical
  if (vertical >= 15 && vertical < 20) {
    if (horizontal >= 45 && horizontal <= 55) {
      return 'neck';
    }
    return 'invalid';
  }

  // Upper back / Shoulders: 20-25% vertical
  if (vertical >= 20 && vertical < 25) {
    if (horizontal >= 37 && horizontal <= 63) {
      return 'upper back';
    }
    return 'invalid';
  }

  // Mid back / Upper arms (back): 25-35% vertical
  if (vertical >= 25 && vertical < 35) {
    // Mid back: 42-58% horizontal
    if (horizontal >= 42 && horizontal <= 58) {
      return 'mid back';
    }
    // Upper arm-L: 35-42% horizontal
    if (horizontal >= 35 && horizontal < 42) {
      return 'upper arm';
    }
    // Upper arm-R: 58-64% horizontal
    if (horizontal > 58 && horizontal <= 64) {
      return 'upper arm';
    }
    return 'invalid';
  }

  // Lower back / Lower arms (back): 35-50% vertical
  if (vertical >= 35 && vertical < 50) {
    // Lower back: 40-60% horizontal
    if (horizontal >= 40 && horizontal <= 60) {
      return 'lower back';
    }
    // Lower arm-L: 33-40% horizontal
    if (horizontal >= 33 && horizontal < 40) {
      return 'lower arm';
    }
    // Lower arm-R: 60-67% horizontal
    if (horizontal > 60 && horizontal <= 67) {
      return 'lower arm';
    }
    return 'invalid';
  }

  // Hands / Buttocks: 50-58% vertical
  if (vertical >= 50 && vertical < 58) {
    // Hands-L: 32-40% horizontal
    if (horizontal >= 32 && horizontal < 40) {
      return 'hands';
    }
    // Hands-R: 60-68% horizontal
    if (horizontal >= 60 && horizontal <= 68) {
      return 'hands';
    }
    // Buttocks: 40-60% horizontal
    if (horizontal >= 40 && horizontal < 60) {
      return 'buttocks';
    }
    return 'invalid';
  }

  // Back of thighs: 58-70% vertical
  if (vertical >= 58 && vertical < 70) {
    // Thigh-L: 40-49% horizontal
    if (horizontal >= 40 && horizontal < 51) {
      return 'thighs';
    }
    // Thigh-R: 51-60% horizontal
    if (horizontal >= 51 && horizontal <= 60) {
      return 'thighs';
    }
    return 'invalid';
  }

  // Calves: 70-90% vertical
  if (vertical >= 70 && vertical < 90) {
    // Calves-L: 40-48% horizontal
    if (horizontal >= 40 && horizontal < 52) {
      return 'calves';
    }
    // Calves-R: 52-59% horizontal
    if (horizontal >= 52 && horizontal <= 59) {
      return 'calves';
    }
    return 'invalid';
  }

  // Back of foot: 90-96% vertical
  if (vertical >= 90 && vertical <= 96) {
    // Foot-R: 38-48% horizontal (left side on back view)
    if (horizontal >= 38 && horizontal < 48) {
      return 'foot';
    }
    // Foot-L: 48-53% horizontal (right side on back view)
    if (horizontal >= 58 && horizontal <= 63) {
      return 'foot';
    }
    return 'invalid';
  }

  return 'invalid';
}

/**
 * Determines body part label based on normalized coordinates (0-100%)
 * Implements bounding box logic for different anatomical regions
 * @param {number} horizontal - Horizontal coordinate as percentage (0-100)
 * @param {number} vertical - Vertical coordinate as percentage (0-100)
 * @param {boolean} isFrontView - Whether viewing front (true) or back (false)
 * @returns {string} Body part label or "invalid"
 */
export function getBodyPartFromCoordinates(horizontal, vertical, isFrontView = true) {
  // If back view, use back body part detection
  if (!isFrontView) {
    return getBackBodyPartFromCoordinates(horizontal, vertical);
  }

  // Front view logic below
  // Overall bounds check - outside body boundaries
  if (vertical < 4 || vertical > 96 || horizontal < 32 || horizontal > 68) {
    return 'invalid';
  }

  // Top region: 4-15% vertical (hair, ear, face)
  if (vertical >= 4 && vertical < 15) {
    // Hair: top center, 45-55% horizontal, 4-8% vertical
    if (vertical >= 4 && vertical < 8 && horizontal >= 45 && horizontal <= 55) {
      return 'hair';
    }
    // Ear-L: 43-45% horizontal, 10-14% vertical
    if (vertical >= 10 && vertical < 14 && horizontal >= 43 && horizontal < 45) {
      return 'ear';
    }
    // Ear-R: 55-57% horizontal, 10-14% vertical
    if (vertical >= 10 && vertical < 14 && horizontal > 55 && horizontal <= 57) {
      return 'ear';
    }
    // Face: center, 45-55% horizontal, 8-15% vertical
    if (vertical >= 8 && horizontal >= 45 && horizontal <= 55) {
      return 'face';
    }
    return 'invalid';
  }

  // Upper region: 15-25% vertical (neck, shoulders)
  if (vertical >= 15 && vertical < 25) {
    // Neck: center, 45-55% horizontal, 15-20% vertical
    if (vertical < 20 && horizontal >= 45 && horizontal <= 55) {
      return 'neck';
    }
    // Shoulders: 37-63% horizontal, 20-25% vertical
    if (vertical >= 20 && horizontal >= 37 && horizontal <= 63) {
      return 'shoulders';
    }
    return 'invalid';
  }

  // Upper-middle region: 25-35% vertical (chest, upper arm, armpit)
  if (vertical >= 25 && vertical < 35) {
    // Armpit-L: 40-42% horizontal, 25-30% vertical
    if (vertical < 30 && horizontal >= 40 && horizontal < 42) {
      return 'armpit';
    }
    // Armpit-R: 58-60% horizontal, 25-30% vertical
    if (vertical < 30 && horizontal > 58 && horizontal <= 60) {
      return 'armpit';
    }
    // Chest: center, 42-58% horizontal
    if (horizontal >= 42 && horizontal <= 58) {
      return 'chest';
    }
    // Upper arm-L: 35-40% horizontal (excluding armpit area)
    if (horizontal >= 35 && horizontal < 40) {
      return 'upper arm';
    }
    // Upper arm-R: 60-64% horizontal (excluding armpit area)
    if (horizontal > 60 && horizontal <= 64) {
      return 'upper arm';
    }
    return 'invalid';
  }

  // Middle region: 35-50% vertical (abdomen, lower arm)
  if (vertical >= 35 && vertical < 50) {
    // Abdomen: center, 40-60% horizontal
    if (horizontal >= 40 && horizontal <= 60) {
      return 'abdomen';
    }
    // Lower arm-L: 33-40% horizontal
    if (horizontal >= 33 && horizontal < 40) {
      return 'lower arm';
    }
    // Lower arm-R: 60-67% horizontal
    if (horizontal > 60 && horizontal <= 67) {
      return 'lower arm';
    }
    return 'invalid';
  }

  // Lower-middle region: 50-70% vertical (hands, hips, groin, thighs)
  if (vertical >= 50 && vertical < 70) {
    // Hands-L: 32-39% horizontal, 50-60% vertical
    if (vertical < 60 && horizontal >= 32 && horizontal < 40) {
      return 'hands';
    }
    // Hands-R: 61-68% horizontal, 50-60% vertical
    if (vertical < 60 && horizontal > 60 && horizontal <= 68) {
      return 'hands';
    }
    
    // Groin: center, 45-55% horizontal, 50-58% vertical
    if (vertical < 58 && horizontal >= 45 && horizontal <= 55) {
      return 'groin';
    }
    // Hip-L: 40-45% horizontal, 50-58% vertical
    if (vertical < 58 && horizontal >= 40 && horizontal < 45) {
      return 'hips';
    }
    // Hip-R: 55-60% horizontal, 50-58% vertical
    if (vertical < 58 && horizontal > 55 && horizontal <= 60) {
      return 'hips';
    }
    
    // Thighs: starting from 58% vertical
    if (vertical >= 58) {
      // Thigh-L: 40-49% horizontal
      if (horizontal >= 40 && horizontal < 51) {
        return 'thighs';
      }
      // Thigh-R: 51-60% horizontal  
      if (horizontal >= 51 && horizontal <= 60) {
        return 'thighs';
      }
    }
    return 'invalid';
  }

  // Lower region: 70-90% vertical (lower legs)
  if (vertical >= 70 && vertical < 90) {
    // Lower legs-L: 40-48% horizontal
    if (horizontal >= 40 && horizontal < 52) {
      return 'lower legs';
    }
    // Lower legs-R: 52-60% horizontal
    if (horizontal >= 52 && horizontal <= 60) {
      return 'lower legs';
    }
    return 'invalid';
  }

  // Bottom region: 90-96% vertical (foot)
  if (vertical >= 90 && vertical <= 96) {
    // Foot-L: 38-48% horizontal
    if (horizontal >= 38 && horizontal < 52) {
      return 'foot';
    }
    // Foot-R: 52-62% horizontal
    if (horizontal >= 52 && horizontal <= 62) {
      return 'foot';
    }
    return 'invalid';
  }

  return 'invalid';
}

// Define bounding boxes for visualization
const BOUNDING_BOXES = [
  // Top region
  { name: 'hair', left: 45, top: 4, width: 10, height: 4, color: '#ff6b6b' },
  { name: 'ear-L', left: 43, top: 10, width: 2, height: 4, color: '#4ecdc4' },
  { name: 'ear-R', left: 55, top: 10, width: 2, height: 4, color: '#4ecdc4' },
  { name: 'face', left: 45, top: 8, width: 10, height: 7, color: '#ffe66d' },
  
  // Upper region
  { name: 'neck', left: 45, top: 15, width: 10, height: 5, color: '#95e1d3' },
  { name: 'shoulders', left: 37, top: 20, width: 26, height: 5, color: '#f38181' },
  
  // Upper-middle region
  { name: 'chest', left: 42, top: 25, width: 16, height: 10, color: '#aa96da' },
  { name: 'upper arm-L', left: 35, top: 25, width: 6, height: 10, color: '#fcbad3' },
  { name: 'upper arm-R', left: 58, top: 25, width: 6, height: 10, color: '#fcbad3' },
  { name: 'upper armpit-R', left: 58, top: 25, width: 2, height: 5, color: '#fcba83' },
  { name: 'upper armpit-L', left: 40, top: 25, width: 2, height: 5, color: '#fcba83' },
  
  // Middle region
  { name: 'abdomen', left: 40, top: 35, width: 20, height: 15, color: '#a8d8ea' },
  { name: 'lower arm-L', left: 33, top: 35, width: 7, height: 15, color: '#ffcccc' },
  { name: 'lower arm-R', left: 60, top: 35, width: 7, height: 15, color: '#ffcccc' },
  
  // Lower-middle region
  { name: 'hands-L', left: 32, top: 50, width: 7, height: 10, color: '#c7ceea' },
  { name: 'hands-R', left: 61, top: 50, width: 7, height: 10, color: '#c7ceea' },
  { name: 'groin', left: 45, top: 50, width: 10, height: 8, color: '#ffc8dd' },
  { name: 'hip-L', left: 40, top: 50, width: 5, height: 8, color: '#ffcc99' },
  { name: 'hip-R', left: 55, top: 50, width: 5, height: 8, color: '#ffcc99' },


  { name: 'thigh-L', left: 40, top: 58, width: 9, height: 12, color: '#bde0fe' },
  { name: 'thigh-R', left: 51, top: 58, width: 9, height: 12, color: '#bde0fe' },
  
  // Lower region
  { name: 'lower legs-L', left: 40, top: 70, width: 8, height: 20, color: '#caffbf' },
  { name: 'lower legs-R', left: 52, top: 70, width: 8, height: 20, color: '#caffbf' },
  
  // Bottom region
  { name: 'foot-L', left: 38, top: 90, width: 10, height: 6, color: '#ffd6a5' },
  { name: 'foot-R', left: 52, top: 90, width: 10, height: 6, color: '#ffd6a5' },
];

// Define bounding boxes for BACK view visualization
const BACK_BOUNDING_BOXES = [
  // Top region - back of head
  { name: 'back of head', left: 45, top: 4, width: 10, height: 11, color: '#ff6b6b' },
  
  // Neck (back)
  { name: 'neck', left: 45, top: 15, width: 10, height: 5, color: '#95e1d3' },
  
  // Upper back / Shoulders
  { name: 'upper back', left: 37, top: 20, width: 26, height: 5, color: '#f38181' },
  
  // Mid back / Upper arms
  { name: 'mid back', left: 42, top: 25, width: 16, height: 10, color: '#aa96da' },
  { name: 'upper arm-L', left: 35, top: 25, width: 7, height: 10, color: '#fcbad3' },
  { name: 'upper arm-R', left: 58, top: 25, width: 6, height: 10, color: '#fcbad3' },
  
  // Lower back / Lower arms
  { name: 'lower back', left: 40, top: 35, width: 20, height: 15, color: '#a8d8ea' },
  { name: 'lower arm-L', left: 33, top: 35, width: 7, height: 15, color: '#ffcccc' },
  { name: 'lower arm-R', left: 60, top: 35, width: 7, height: 15, color: '#ffcccc' },
  
  // Hands / Buttocks
  { name: 'hands-L', left: 32, top: 50, width: 8, height: 8, color: '#c7ceea' },
  { name: 'hands-R', left: 60, top: 50, width: 8, height: 8, color: '#c7ceea' },
  { name: 'buttocks', left: 40, top: 50, width: 20, height: 8, color: '#ffc8dd' },
  
  // Thighs (back)
  { name: 'thigh-L', left: 40, top: 58, width: 9, height: 12, color: '#bde0fe' },
  { name: 'thigh-R', left: 51, top: 58, width: 9, height: 12, color: '#bde0fe' },
  
  // Calves
  { name: 'calves-L', left: 40, top: 70, width: 8, height: 20, color: '#caffbf' },
  { name: 'calves-R', left: 52, top: 70, width: 7, height: 20, color: '#caffbf' },
  
  // Foot (back)
  { name: 'foot-L', left: 48, top: 90, width: 5, height: 6, color: '#ffd6a5' },
  { name: 'foot-R', left: 38, top: 90, width: 10, height: 6, color: '#ffd6a5' },
];

// Renders a responsive body image and reports normalized coordinates (percent)
export default function BodyMapPicker({ value, onChange, imgSrc = '/assets/human_body.png', imgAlt = 'Body Map', showBoundingBoxes = false }) {
  const imgRef = useRef(null);
  const [pos, setPos] = useState(value || null);
  const [hoveredBox, setHoveredBox] = useState(null);
  const [isFrontView, setIsFrontView] = useState(true);

  useEffect(() => setPos(value || null), [value]);

  const toggleView = () => {
    setIsFrontView(!isFrontView);
    // Clear position when switching views
    setPos(null);
    if (onChange) onChange(null);
  };

  const currentImgSrc = isFrontView ? '/assets/human_body.png' : '/assets/back.png';

  const handlePointer = (clientX, clientY) => {
    const img = imgRef.current;
    if (!img) return;
    const rect = img.getBoundingClientRect();
    const left = (clientX - rect.left) / rect.width;
    const top = (clientY - rect.top) / rect.height;
    const leftPct = Math.max(0, Math.min(1, left)) * 100;
    const topPct = Math.max(0, Math.min(1, top)) * 100;
    const bodyPart = getBodyPartFromCoordinates(leftPct, topPct, isFrontView);
    const newPos = { leftPct, topPct, bodyPart, isFrontView };
    setPos(newPos);
    onChange && onChange(newPos);
  };

  const handleClick = (e) => {
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    handlePointer(clientX, clientY);
  };

  return (
    <Box sx={{ position: 'relative', width: '100%' }}>
      {/* Flip button */}
      <Button
        variant="outlined"
        size="small"
        startIcon={<FlipIcon />}
        onClick={toggleView}
        sx={{
          position: 'absolute',
          top: 8,
          right: 8,
          zIndex: 20,
          bgcolor: 'white',
          '&:hover': { bgcolor: 'rgba(255,255,255,0.9)' },
        }}
      >
        {isFrontView ? 'Show Back' : 'Show Front'}
      </Button>

      <Box sx={{ position: 'relative', width: '100%', touchAction: 'manipulation' }}>
        <img
          ref={imgRef}
          src={currentImgSrc}
          alt={`${imgAlt} - ${isFrontView ? 'Front' : 'Back'} View`}
          style={{ width: '100%', height: 'auto', display: 'block', userSelect: 'none' }}
          onClick={handleClick}
          onTouchStart={handleClick}
        />

      {/* Bounding boxes overlay */}
      {showBoundingBoxes && (isFrontView ? BOUNDING_BOXES : BACK_BOUNDING_BOXES).map((box, idx) => (
        <Box
          key={idx}
          onMouseEnter={() => setHoveredBox(box.name)}
          onMouseLeave={() => setHoveredBox(null)}
          sx={{
            position: 'absolute',
            left: `${box.left}%`,
            top: `${box.top}%`,
            width: `${box.width}%`,
            height: `${box.height}%`,
            border: `2px solid ${box.color}`,
            backgroundColor: hoveredBox === box.name ? `${box.color}33` : `${box.color}11`,
            pointerEvents: 'none',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '10px',
            fontWeight: 'bold',
            color: box.color,
            textShadow: '0 0 3px white, 0 0 3px white',
            transition: 'background-color 0.2s',
          }}
        >
          {hoveredBox === box.name && box.name}
        </Box>
      ))}

      {/* Selected position marker */}
      {pos && (
        <Box
          sx={{
            position: 'absolute',
            left: `${pos.leftPct}%`,
            top: `${pos.topPct}%`,
            transform: 'translate(-50%,-50%)',
            width: 18,
            height: 18,
            borderRadius: '50%',
            bgcolor: 'primary.main',
            border: '2px solid #fff',
            boxShadow: 2,
            pointerEvents: 'none',
            zIndex: 10,
          }}
        />
      )}

      {/* Coordinates display */}
      {pos && showBoundingBoxes && (
        <Box
          sx={{
            position: 'absolute',
            bottom: 10,
            left: '50%',
            transform: 'translateX(-50%)',
            bgcolor: 'rgba(0,0,0,0.7)',
            color: 'white',
            px: 2,
            py: 1,
            borderRadius: 1,
            fontSize: '12px',
            fontFamily: 'monospace',
            pointerEvents: 'none',
          }}
        >
          H: {pos.leftPct.toFixed(1)}% | V: {pos.topPct.toFixed(1)}% | {getBodyPartFromCoordinates(pos.leftPct, pos.topPct, isFrontView)}
        </Box>
      )}
      </Box>
    </Box>
  );
}
