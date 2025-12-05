'use client';

import { useRef, useState, useEffect } from 'react';
import { Box } from '@mui/material';

/**
 * Determines body part label based on normalized coordinates (0-100%)
 * Implements bounding box logic for different anatomical regions
 * @param {number} horizontal - Horizontal coordinate as percentage (0-100)
 * @param {number} vertical - Vertical coordinate as percentage (0-100)
 * @returns {string} Body part label or "invalid"
 */
export function getBodyPartFromCoordinates(horizontal, vertical) {
  // Overall bounds check
  if (vertical < 2 || vertical > 97 || horizontal < 30 || horizontal > 70) {
    return 'invalid';
  }

  // Top region: 2-15% vertical (hair, ear, face)
  if (vertical >= 2 && vertical < 15) {
    // Hair: top center, 45-55% horizontal
    if (vertical >= 2 && vertical < 8 && horizontal >= 45 && horizontal <= 55) {
      return 'hair';
    }
    // Ear: sides, 35-45% or 55-65% horizontal
    if (horizontal >= 35 && horizontal < 45) {
      return 'ear';
    }
    if (horizontal > 55 && horizontal <= 65) {
      return 'ear';
    }
    // Face: center, 40-60% horizontal
    if (horizontal >= 40 && horizontal <= 60) {
      return 'face';
    }
    return 'invalid';
  }

  // Upper region: 15-25% vertical (neck, shoulders)
  if (vertical >= 15 && vertical < 25) {
    // Neck: center, 45-55% horizontal
    if (horizontal >= 45 && horizontal <= 55) {
      return 'neck';
    }
    // Shoulders: wider area, 35-65% horizontal
    if (horizontal >= 35 && horizontal <= 65) {
      return 'shoulders';
    }
    return 'invalid';
  }

  // Upper-middle region: 25-35% vertical (chest, upper arm)
  if (vertical >= 25 && vertical < 35) {
    // Chest: center, 40-60% horizontal
    if (horizontal >= 40 && horizontal <= 60) {
      return 'chest';
    }
    // Upper arm: sides, 30-40% or 60-70% horizontal
    if ((horizontal >= 30 && horizontal < 40) || (horizontal > 60 && horizontal <= 70)) {
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
    // Lower arm: sides, 30-40% or 60-70% horizontal
    if ((horizontal >= 30 && horizontal < 40) || (horizontal > 60 && horizontal <= 70)) {
      return 'lower arm';
    }
    return 'invalid';
  }

  // Lower-middle region: 50-70% vertical (hands, groin, thighs)
  if (vertical >= 50 && vertical < 70) {
    // Hands: sides, 30-40% or 60-70% horizontal, upper part
    if (vertical >= 50 && vertical < 56) {
      if ((horizontal >= 30 && horizontal < 40) || (horizontal > 60 && horizontal <= 70)) {
        return 'hands';
      }
    }
    // Groin: center, 45-55% horizontal, covers V part (extends down to prevent center gap)
    if (vertical >= 56 && vertical < 66 && horizontal >= 45 && horizontal <= 55) {
      return 'groin';
    }
    // Thighs: left and right bounding boxes, lower part (excluding center area)
    if (vertical >= 62 && vertical < 70) {
      // Left thigh: 40-45% horizontal (not including center)
      if (horizontal >= 40 && horizontal < 45) {
        return 'thighs';
      }
      // Right thigh: 55-60% horizontal (not including center)
      if (horizontal > 55 && horizontal <= 60) {
        return 'thighs';
      }
    }
    return 'invalid';
  }

  // Lower region: 70-90% vertical (lower legs)
  if (vertical >= 70 && vertical < 90) {
    // Lower legs: center area, 40-60% horizontal
    if (horizontal >= 40 && horizontal <= 60) {
      return 'lower legs';
    }
    return 'invalid';
  }

  // Bottom region: 90-97% vertical (foot)
  if (vertical >= 90 && vertical <= 97) {
    // Foot: center area, 40-60% horizontal
    if (horizontal >= 40 && horizontal <= 60) {
      return 'foot';
    }
    return 'invalid';
  }

  return 'invalid';
}

// Renders a responsive body image and reports normalized coordinates (percent)
export default function BodyMapPicker({ value, onChange, imgSrc = '/assets/human_body.png', imgAlt = 'Body Map' }) {
  const imgRef = useRef(null);
  const [pos, setPos] = useState(value || null);

  useEffect(() => setPos(value || null), [value]);

  const handlePointer = (clientX, clientY) => {
    const img = imgRef.current;
    if (!img) return;
    const rect = img.getBoundingClientRect();
    const left = (clientX - rect.left) / rect.width;
    const top = (clientY - rect.top) / rect.height;
    const leftPct = Math.max(0, Math.min(1, left)) * 100;
    const topPct = Math.max(0, Math.min(1, top)) * 100;
    const newPos = { leftPct, topPct };
    setPos(newPos);
    onChange && onChange(newPos);
  };

  const handleClick = (e) => {
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    handlePointer(clientX, clientY);
  };

  return (
    <Box sx={{ position: 'relative', width: '100%', touchAction: 'manipulation' }}>
      <img
        ref={imgRef}
        src={imgSrc}
        alt={imgAlt}
        style={{ width: '100%', height: 'auto', display: 'block', userSelect: 'none' }}
        onClick={handleClick}
        onTouchStart={handleClick}
      />

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
          }}
        />
      )}
    </Box>
  );
}
