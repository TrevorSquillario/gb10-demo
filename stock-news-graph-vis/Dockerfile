FROM node:20-alpine

WORKDIR /app

# Install dependencies first (layer cache)
COPY package.json ./
RUN npm install --omit=dev

# Copy application files
COPY server.js ./
COPY public/ ./public/

EXPOSE 3500

CMD ["node", "server.js"]
