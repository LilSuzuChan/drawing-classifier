FROM node:22-alpine AS build

WORKDIR /app

COPY web/package*.json ./
RUN npm ci

COPY web/ ./

RUN npm run build


FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html

# nginxのデフォルトmime.typesには.mjsが無く、ブラウザがESモジュールとして
# 実行を拒否してしまう(onnxruntime-webのWASMローダーが.mjsを動的importする)ため追加する
RUN printf 'types {\n    application/javascript mjs;\n}\n' > /etc/nginx/conf.d/mjs-mime.conf

COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
