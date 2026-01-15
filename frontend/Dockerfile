# Use Node.js for the build and development environment
FROM node:20-slim

WORKDIR /app

# Copy package files
COPY package*.json ./


# Install dependencies
COPY package-lock.json ./
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi

# Copy source code
COPY . .

# Expose the default Vite port
EXPOSE 5173

# Start the development server
# --host is required to allow access from outside the container
CMD ["npm", "run", "dev", "--", "--host"]
