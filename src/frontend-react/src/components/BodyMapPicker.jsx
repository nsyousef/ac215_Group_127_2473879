'use client';

import { useRef, useState, useEffect } from 'react';
import { Box } from '@mui/material';

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
