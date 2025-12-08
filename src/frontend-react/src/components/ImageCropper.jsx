'use client';

import { useState, useRef, useEffect } from 'react';
import { Box, Button, Typography } from '@mui/material';

export default function ImageCropper({ imageSrc, onCropComplete, onCancel }) {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPos, setStartPos] = useState({ x: 0, y: 0 });
  const [currentPos, setCurrentPos] = useState({ x: 0, y: 0 });
  const [cropBox, setCropBox] = useState(null);
  const [imageLoaded, setImageLoaded] = useState(false);

  useEffect(() => {
    const img = new Image();
    img.onload = () => {
      imageRef.current = img;
      setImageLoaded(true);
      drawCanvas();
    };
    img.src = imageSrc;
  }, [imageSrc]);

  useEffect(() => {
    if (imageLoaded) {
      drawCanvas();
    }
  }, [cropBox, imageLoaded]);

  const drawCanvas = () => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    if (!canvas || !img) return;

    const ctx = canvas.getContext('2d');
    const maxWidth = canvas.parentElement.clientWidth - 40;
    const maxHeight = 500;

    let scale = Math.min(maxWidth / img.width, maxHeight / img.height, 1);
    const displayWidth = img.width * scale;
    const displayHeight = img.height * scale;

    canvas.width = displayWidth;
    canvas.height = displayHeight;

    // Draw image
    ctx.drawImage(img, 0, 0, displayWidth, displayHeight);

    // Draw crop box if exists
    if (cropBox) {
      const { x, y, width, height } = cropBox;
      
      // Darken everything outside crop box
      ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.clearRect(x, y, width, height);
      ctx.drawImage(img, 
        x / scale, y / scale, width / scale, height / scale,
        x, y, width, height
      );

      // Draw border
      ctx.strokeStyle = '#0891B2';
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, width, height);

      // Draw corner handles
      const handleSize = 10;
      ctx.fillStyle = '#0891B2';
      [
        [x, y], [x + width, y], 
        [x, y + height], [x + width, y + height]
      ].forEach(([hx, hy]) => {
        ctx.fillRect(hx - handleSize / 2, hy - handleSize / 2, handleSize, handleSize);
      });
    }
  };

  const getMousePos = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  };

  const handleMouseDown = (e) => {
    const pos = getMousePos(e);
    setStartPos(pos);
    setCurrentPos(pos);
    setIsDrawing(true);
    setCropBox(null);
  };

  const handleMouseMove = (e) => {
    if (!isDrawing) return;
    const pos = getMousePos(e);
    setCurrentPos(pos);
    
    const x = Math.min(startPos.x, pos.x);
    const y = Math.min(startPos.y, pos.y);
    const width = Math.abs(pos.x - startPos.x);
    const height = Math.abs(pos.y - startPos.y);
    
    setCropBox({ x, y, width, height });
  };

  const handleMouseUp = () => {
    setIsDrawing(false);
  };

  const handleCrop = () => {
    if (!cropBox || !imageRef.current) return;

    const canvas = canvasRef.current;
    const img = imageRef.current;
    const scale = canvas.width / img.width;

    // Calculate crop coordinates in original image space
    const cropData = {
      x: Math.round(cropBox.x / scale),
      y: Math.round(cropBox.y / scale),
      width: Math.round(cropBox.width / scale),
      height: Math.round(cropBox.height / scale),
    };

    // Create cropped image on a new canvas
    const cropCanvas = document.createElement('canvas');
    cropCanvas.width = cropData.width;
    cropCanvas.height = cropData.height;
    const cropCtx = cropCanvas.getContext('2d');
    
    cropCtx.drawImage(
      img,
      cropData.x, cropData.y, cropData.width, cropData.height,
      0, 0, cropData.width, cropData.height
    );

    // Convert to blob and data URL
    cropCanvas.toBlob((blob) => {
      const croppedDataUrl = cropCanvas.toDataURL('image/jpeg', 0.95);
      onCropComplete(croppedDataUrl, cropData, blob);
    }, 'image/jpeg', 0.95);
  };

  const handleSkip = () => {
    // User chose not to crop - use full image
    const img = imageRef.current;
    if (!img) return;
    
    const cropData = {
      x: 0,
      y: 0,
      width: img.width,
      height: img.height,
    };
    
    onCropComplete(imageSrc, cropData, null);
  };

  return (
    <Box>
      <Typography variant="subtitle1" sx={{ mb: 1 }}>
        Crop Image (Optional)
      </Typography>
      <Typography variant="body2" sx={{ color: '#666', mb: 2 }}>
        Draw a box around the area you want to analyze, or skip to use the full image.
      </Typography>

      <Box
        sx={{
          border: '2px solid #ddd',
          borderRadius: 1,
          overflow: 'hidden',
          mb: 2,
          display: 'flex',
          justifyContent: 'center',
          bgcolor: '#f5f5f5',
        }}
      >
        <canvas
          ref={canvasRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          style={{
            cursor: isDrawing ? 'crosshair' : 'crosshair',
            maxWidth: '100%',
            display: 'block',
          }}
        />
      </Box>

      <Box sx={{ display: 'flex', gap: 1, justifyContent: 'center' }}>
        <Button
          variant="outlined"
          onClick={handleSkip}
          size="small"
        >
          Skip Cropping
        </Button>
        <Button
          variant="contained"
          onClick={handleCrop}
          disabled={!cropBox}
          size="small"
        >
          Apply Crop
        </Button>
      </Box>
    </Box>
  );
}

