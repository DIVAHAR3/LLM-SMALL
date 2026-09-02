import { useState } from 'react'
import { generateTextStream } from './api'
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

  function appendToLastMessage(chunk) {
    setMessages((prev) => {
      const updated = [...prev]
      const last = updated[updated.length - 1]
      updated[updated.length - 1] = { ...last, text: last.text + chunk }
      return updated
    })
  }

  async function handleSend(event) {
    event.preventDefault()
    const prompt = input.trim()
    if (!prompt || loading) return

    setMessages((prev) => [...prev, { role: 'user', text: prompt }, { role: 'assistant', text: '' }])
    setInput('')
    setLoading(true)
    setError(null)

    try {
      await generateTextStream(prompt, { max_new_tokens: 150, temperature: 0.8 }, appendToLastMessage)
    } catch (err) {
      setError(err.message)
      // Drop the placeholder bubble only if nothing ever streamed into it;
      // if some text arrived before the failure, keep it -- it's real output.
      setMessages((prev) => {
        const last = prev[prev.length - 1]
        return last && last.role === 'assistant' && last.text === '' ? prev.slice(0, -1) : prev
      })
    } finally {
      setLoading(false)
    }
  }

  function handleClear() {
    setMessages([])
    setError(null)
  }

  const lastMessage = messages[messages.length - 1]
  const isAwaitingFirstChunk = loading && lastMessage?.role === 'assistant' && lastMessage.text === ''

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
        {messages.length === 0 && (
          <p className="empty-state">Send a prompt to see what the model generates.</p>
        )}
        {messages.map((message, index) => (
          <MessageBubble
            key={index}
            role={message.role}
            text={index === messages.length - 1 && isAwaitingFirstChunk ? '…' : message.text}
          />
        ))}
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
