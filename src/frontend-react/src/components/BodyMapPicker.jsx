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

  // Head: 2-19% vertical, 41-59% horizontal
  if (vertical >= 2 && vertical < 19) {
    if (horizontal >= 41 && horizontal <= 59) {
      return 'head';
    }
    return 'invalid';
  }

  // Torso and Arms: 19-54% vertical
  if (vertical >= 19 && vertical < 54) {
    // Torso: 41-59% horizontal
    if (horizontal >= 41 && horizontal <= 59) {
      return 'torso';
    }

    // Left side: 30-41% horizontal
    if (horizontal >= 30 && horizontal < 41) {
      if (vertical >= 19 && vertical < 38) {
        return 'left upper arm';
      }
      if (vertical >= 38 && vertical < 54) {
        return 'left lower arm';
      }
    }

    // Right side: 59-70% horizontal
    if (horizontal > 59 && horizontal <= 70) {
      if (vertical >= 19 && vertical < 38) {
        return 'right upper arm';
      }
      if (vertical >= 38 && vertical < 54) {
        return 'right lower arm';
      }
    }

    return 'invalid';
  }

  // Hands and Upper Legs: 54-68% vertical
  if (vertical >= 54 && vertical < 68) {
    // Left hand: 30-41% horizontal
    if (horizontal >= 30 && horizontal < 41) {
      if (vertical >= 54 && vertical < 61) {
        return 'left hand';
      }
      return 'invalid'; // >= 61%
    }

    // Left upper leg: 41-50% horizontal
    if (horizontal >= 41 && horizontal < 50) {
      return 'left upper leg';
    }

    // Right upper leg: 50-59% horizontal
    if (horizontal >= 50 && horizontal <= 59) {
      return 'right upper leg';
    }

    // Right hand: 59-70% horizontal
    if (horizontal > 59 && horizontal <= 70) {
      if (vertical >= 54 && vertical < 61) {
        return 'right hand';
      }
      return 'invalid'; // >= 61%
    }

    return 'invalid';
  }

  // Lower Legs: 68-88% vertical
  if (vertical >= 68 && vertical < 88) {
    // Left lower leg: 41-50% horizontal
    if (horizontal >= 41 && horizontal < 50) {
      return 'left lower leg';
    }

    // Right lower leg: 50-59% horizontal
    if (horizontal >= 50 && horizontal <= 59) {
      return 'right lower leg';
    }

    return 'invalid';
  }

  // Feet: 88-97% vertical
  if (vertical >= 88 && vertical <= 97) {
    // Left foot: 41-50% horizontal
    if (horizontal >= 41 && horizontal < 50) {
      return 'left foot';
    }

    // Right foot: 50-59% horizontal
    if (horizontal >= 50 && horizontal <= 59) {
      return 'right foot';
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
