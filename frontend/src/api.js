import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'
const IS_DEV = import.meta.env.MODE === 'development'
const API_KEY = import.meta.env.VITE_API_KEY || (IS_DEV ? 'default_secret_key' : undefined)

const apiHeaders = {
  Accept: 'application/json',
}
if (API_KEY) {
  apiHeaders['X-API-KEY'] = API_KEY
}

const api = axios.create({
  baseURL: API_BASE,
  headers: apiHeaders,
})

export default api
