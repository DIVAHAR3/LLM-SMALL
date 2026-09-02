import { useState } from 'react'
import { generateText } from './api'
import './App.css'

function MessageBubble({ role, text }) {
  return (
    <div className={`bubble ${role}`}>
      <span className="bubble-label">{role === 'user' ? 'You' : 'Model'}</span>
      <p>{text}</p>
    </div>
  )
}

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handleSend(event) {
    event.preventDefault()
    const prompt = input.trim()
    if (!prompt || loading) return

    setMessages((prev) => [...prev, { role: 'user', text: prompt }])
    setInput('')
    setLoading(true)
    setError(null)

    try {
      const text = await generateText(prompt, { max_new_tokens: 100, temperature: 0.8 })
      setMessages((prev) => [...prev, { role: 'assistant', text }])
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  function handleClear() {
    setMessages([])
    setError(null)
  }

  return (
    <div className="chat-app">
      <header>
        <div>
          <h1>GPT-from-Scratch</h1>
          <p className="subtitle">A tiny, self-trained model — expect pattern-matching, not coherent conversation.</p>
        </div>
        <button type="button" className="clear-button" onClick={handleClear} disabled={messages.length === 0}>
          Clear
        </button>
      </header>

      <div className="message-list">
        {messages.length === 0 && !loading && (
          <p className="empty-state">Send a prompt to see what the model generates.</p>
        )}
        {messages.map((message, index) => (
          <MessageBubble key={index} role={message.role} text={message.text} />
        ))}
        {loading && (
          <div className="bubble assistant loading">
            <span className="bubble-label">Model</span>
            <p>Generating…</p>
          </div>
        )}
      </div>

      {error && <div className="error-banner">Error: {error}</div>}

      <form className="chat-input" onSubmit={handleSend}>
        <input
          type="text"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="Type a prompt…"
          disabled={loading}
        />
        <button type="submit" disabled={loading || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  )
}

export default App
