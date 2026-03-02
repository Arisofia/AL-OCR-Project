# Stage 1: Build
FROM node:20-slim AS build

WORKDIR /app


COPY package*.json ./
RUN if [ -f package-lock.json ]; then npm ci; fi
RUN if [ ! -f package-lock.json ]; then npm install; fi

COPY . .
# Vite will pick up .env.production automatically during build
RUN npm run build

# Stage 2: Production
FROM nginx:stable-alpine

# Add a non-root user
RUN addgroup -S appgroup && adduser -S appuser -G appgroup

# Set permissions for nginx to run as non-root
RUN touch /var/run/nginx.pid && \
    chown -R appuser:appgroup /var/run/nginx.pid /var/cache/nginx /var/log/nginx /etc/nginx/conf.d

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY certs/ /etc/nginx/certs/

# Ensure permissions for content
RUN chown -R appuser:appgroup /usr/share/nginx/html /etc/nginx/certs

# Switch to non-root user
USER appuser

EXPOSE 8080 8443
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD pgrep nginx || exit 1
CMD ["nginx", "-g", "daemon off;"]
