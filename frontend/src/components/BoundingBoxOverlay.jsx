import { useRef, useEffect } from 'react';

const BoundingBoxOverlay = ({ detections, mediaWidth, mediaHeight, displayedWidth, displayedHeight }) => {
    const canvasRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || !detections) return;

        const ctx = canvas.getContext('2d');

        // Set logical size equal to displayed size
        canvas.width = displayedWidth;
        canvas.height = displayedHeight;

        // Clear previous frame
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (mediaWidth === 0 || mediaHeight === 0) return;

        // Scale Calculation:
        const scaleX = displayedWidth / mediaWidth;
        const scaleY = displayedHeight / mediaHeight;

        detections.forEach(det => {
            // Support both Object {x1,y1...} and Array [x1,y1,x2,y2]
            let x1, y1, x2, y2;

            if (Array.isArray(det.bbox)) {
                // Backend V2 Format: [x1, y1, x2, y2]
                [x1, y1, x2, y2] = det.bbox;
            } else if (det.bbox && typeof det.bbox === 'object') {
                // Backend V1 Format: {x1, y1, x2, y2}
                x1 = det.bbox.x1;
                y1 = det.bbox.y1;
                x2 = det.bbox.x2;
                y2 = det.bbox.y2;
            } else if (det.box) {
                // Fallback
                x1 = det.box.x1;
                y1 = det.box.y1;
                x2 = det.box.x2;
                y2 = det.box.y2;
            } else {
                return;
            }

            const x = x1 * scaleX;
            const y = y1 * scaleY;
            const w = (x2 - x1) * scaleX;
            const h = (y2 - y1) * scaleY;

            // --- Theme & Color Logic ---
            const type = det.type || "unknown"; // "safe" or "unsafe"
            const labelLower = (det.label || "unknown").toLowerCase();
            let color = '#3b82f6'; // Default Blue

            // Priority: Type -> Label
            if (type === 'unsafe') {
                color = '#ef4444'; // Red for UNSAFE
            } else if (type === 'safe') {
                color = '#10b981'; // Green for SAFE
            } else {
                // Infer from label if type missing
                if (['fire', 'smoke', 'fighting', 'fight', 'crash'].some(k => labelLower.includes(k))) color = '#ef4444';
                else if (['vehicle', 'car', 'bus', 'truck'].some(k => labelLower.includes(k))) color = '#f97316';
                else color = '#3b82f6';
            }

            // Draw Box
            ctx.shadowBlur = 0;
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.strokeRect(x, y, w, h);

            // Label Formatting
            const confVal = det.confidence !== undefined ? det.confidence : 0;
            const labelText = `${det.label.toUpperCase()} ${(confVal * 100).toFixed(0)}%`;

            ctx.font = 'bold 12px "Inter", sans-serif';
            const textMetrics = ctx.measureText(labelText);
            const bgW = textMetrics.width + 12;
            const bgH = 22;

            // Draw Label Background
            const labelY = y > 24 ? y - 24 : y;

            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.roundRect(x, labelY, bgW, bgH, [4, 4, 4, 4]);
            ctx.fill();

            // Draw Text
            ctx.fillStyle = '#ffffff';
            ctx.fillText(labelText, x + 6, labelY + 15);
        });

    }, [detections, mediaWidth, mediaHeight, displayedWidth, displayedHeight]);

    return (
        <canvas
            ref={canvasRef}
            className="absolute top-0 left-0 pointer-events-none"
            style={{
                width: displayedWidth,
                height: displayedHeight,
                display: 'block'
            }}
        />
    );
};

export default BoundingBoxOverlay;
