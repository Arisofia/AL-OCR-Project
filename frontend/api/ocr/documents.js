import proxyHandler from "../[...path].js";

export default async function handler(req, res) {
  req.query = { ...req.query, path: ["ocr", "documents"] };
  return proxyHandler(req, res);
}
