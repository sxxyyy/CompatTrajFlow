#!/usr/bin/env node
/**
 * Render POI (Point of Interest) view PNGs from trajectory JSON files.
 *
 * Adapted from Traj-MLLM's 3_vis_poi.js.
 *
 * Usage:
 *   node render_poi.js --json-dir <dir> --image-dir <dir> [--html-dir <dir>]
 *
 * Environment variables (fallback when flags are omitted):
 *   TRAJ_MLLM_JSON_DIR   – directory containing trajectory JSON files
 *   TRAJ_MLLM_IMAGE_DIR  – directory where PNG screenshots are saved
 *   TRAJ_MLLM_HTML_DIR   – directory where intermediate HTML files are saved
 */

const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer');

function parseArgs() {
    const args = process.argv.slice(2);
    const opts = {
        jsonDir: process.env.TRAJ_MLLM_JSON_DIR || '../data/ano_trajectory_jsons_poi',
        imageDir: process.env.TRAJ_MLLM_IMAGE_DIR || '../data/poi_anomaly_images',
        htmlDir: process.env.TRAJ_MLLM_HTML_DIR || '../data/poi_anomaly_htmls',
        concurrency: Math.min(
            parseInt(process.env.TRAJ_MLLM_CONCURRENCY || '4', 10), 16
        ),
    };
    for (let i = 0; i < args.length; i++) {
        if (args[i] === '--json-dir' && i + 1 < args.length) {
            opts.jsonDir = args[++i];
        } else if (args[i] === '--image-dir' && i + 1 < args.length) {
            opts.imageDir = args[++i];
        } else if (args[i] === '--html-dir' && i + 1 < args.length) {
            opts.htmlDir = args[++i];
        } else if ((args[i] === '--concurrency' || args[i] === '-c') && i + 1 < args.length) {
            opts.concurrency = Math.min(parseInt(args[++i], 10) || 4, 16);
        }
    }
    return opts;
}

const opts = parseArgs();

if (!fs.existsSync(opts.imageDir)) {
    fs.mkdirSync(opts.imageDir, { recursive: true });
}
if (!fs.existsSync(opts.htmlDir)) {
    fs.mkdirSync(opts.htmlDir, { recursive: true });
}

const createHtmlTemplate = (trajectory) => {
    const lineArr = trajectory.o_geo || [];
    if (lineArr.length < 2) {
        console.error('Warning: Insufficient trajectory coordinate points');
    }
    const trajectoryPoints = JSON.stringify(lineArr);

    return `<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="initial-scale=1.0, user-scalable=no, width=device-width">
    <title>Trajectory - ${trajectory.devid || 'Unknown'}</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        html, body, #map { height: 100%; width: 100%; margin: 0; padding: 0; }
        .info { position: absolute; top: 10px; left: 60px; background: rgba(255,255,255,0.8);
                padding: 10px; border-radius: 5px; z-index: 1000; }
    </style>
</head>
<body>
    <div id="map"></div>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        // Initial view as fallback; fitBounds will override once trajectory is set.
        const map = L.map('map').setView([41.15, -8.61], 13);
        window.mapFullyLoaded = false;

        const tileLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }).addTo(map);

        const trajectoryData = ${trajectoryPoints};
        const fullPathLine = L.polyline([], { color: 'Red', weight: 6, opacity: 0.6 }).addTo(map);

        const startIcon = L.divIcon({
            html: '<div style="width:30px;height:30px;border-radius:50%;background:#4CAF50;color:white;display:flex;justify-content:center;align-items:center;font-weight:bold;font-size:14px;">S</div>',
            className: 'custom-start-icon', iconSize: [30, 30], iconAnchor: [15, 15]
        });
        const endIcon = L.divIcon({
            html: '<div style="width:30px;height:30px;border-radius:50%;background:#F44336;color:white;display:flex;justify-content:center;align-items:center;font-weight:bold;font-size:14px;">E</div>',
            className: 'custom-end-icon', iconSize: [30, 30], iconAnchor: [15, 15]
        });

        if (trajectoryData.length > 0) {
            const points = trajectoryData.map(point => [point[1], point[0]]);
            fullPathLine.setLatLngs(points);
            // Fit without animation so the view is settled before screenshot.
            map.fitBounds(fullPathLine.getBounds(), { padding: [60, 60], animate: false });
            map.whenReady(function() {
                L.marker(points[0], {icon: startIcon}).addTo(map);
                L.marker(points[points.length - 1], {icon: endIcon}).addTo(map);
                window.mapFullyLoaded = true;
            });
        } else {
            window.mapFullyLoaded = true;
        }
    </script>
</body>
</html>`;
};

async function renderAndCaptureTrajectory(browser, trajectoryFilePath) {
    try {
        const trajectoryData = JSON.parse(fs.readFileSync(trajectoryFilePath, 'utf8'));
        const deviceId = path.basename(trajectoryFilePath, '.json');

        const imagePath = path.join(opts.imageDir, `${deviceId}.png`);
        if (fs.existsSync(imagePath)) {
            return { imagePath, skipped: true };
        }

        const htmlContent = createHtmlTemplate(trajectoryData);
        const htmlFilePath = path.join(opts.htmlDir, `${deviceId}.html`);
        fs.writeFileSync(htmlFilePath, htmlContent);

        const page = await browser.newPage();
        try {
            await page.setViewport({ width: 1920, height: 1920 });

            await page.goto(`file://${path.resolve(htmlFilePath)}`, {
                waitUntil: 'networkidle2', timeout: 60000
            });

            await new Promise(resolve => setTimeout(resolve, 2000));

            // Clip to trajectory bounds (with generous padding).
            const clipRect = await page.evaluate(() => {
                const bounds = fullPathLine.getBounds();
                if (!bounds || !bounds.isValid()) return null;
                const nw = map.latLngToContainerPoint(bounds.getNorthWest());
                const se = map.latLngToContainerPoint(bounds.getSouthEast());
                const pad = 80;
                return {
                    x: Math.max(0, nw.x - pad),
                    y: Math.max(0, nw.y - pad),
                    width: Math.min(1920, se.x - nw.x + 2 * pad),
                    height: Math.min(1920, se.y - nw.y + 2 * pad)
                };
            });
            const screenshotOpts = clipRect
                ? { path: imagePath, clip: clipRect }
                : { path: imagePath };
            await page.screenshot(screenshotOpts);
        } finally {
            await page.close();
        }
        return { imagePath };
    } catch (error) {
        console.error('Error rendering trajectory:', error);
        return null;
    }
}

async function main() {
    if (!fs.existsSync(opts.jsonDir)) {
        console.error('JSON directory not found:', opts.jsonDir);
        process.exit(1);
    }

    const files = fs.readdirSync(opts.jsonDir)
        .filter(f => f.endsWith('.json'))
        .sort((a, b) => (parseInt(a, 10) || 0) - (parseInt(b, 10) || 0));
    if (files.length === 0) {
        console.log('No trajectory JSON files found.');
        return;
    }

    console.log(`Found ${files.length} trajectory files in ${opts.jsonDir}`);
    console.log(`Concurrency: ${opts.concurrency} pages`);

    const browser = await puppeteer.launch({
        executablePath: process.env.PUPPETEER_EXECUTABLE_PATH || '/usr/bin/google-chrome',
        args: [
            '--allow-file-access-from-files',
            '--disable-web-security',
            '--disable-features=IsolateOrigins,site-per-process',
            '--no-sandbox',
            '--disable-setuid-sandbox',
        ]
    });

    try {
        let rendered = 0;
        let skipped = 0;
        let failed = 0;
        for (let i = 0; i < files.length; i += opts.concurrency) {
            const chunk = files.slice(i, i + opts.concurrency);
            const tasks = chunk.map((file) => {
                const filePath = path.join(opts.jsonDir, file);
                return renderAndCaptureTrajectory(browser, filePath);
            });
            const results = await Promise.all(tasks);
            for (const result of results) {
                if (result) {
                    if (result.skipped) { skipped++; }
                    else { rendered++; }
                } else {
                    failed++;
                }
            }
            const done = Math.min(i + opts.concurrency, files.length);
            console.log(`Progress: ${done}/${files.length} (${rendered} new, ${skipped} cached, ${failed} failed)`);
        }
        console.log(`Done. Rendered: ${rendered}, Skipped: ${skipped}, Failed: ${failed}`);
    } finally {
        await browser.close();
    }
}

main().catch(err => { console.error(err); process.exit(1); });
