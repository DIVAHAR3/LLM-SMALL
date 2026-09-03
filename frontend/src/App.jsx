import { useState } from 'react'
import { analyzeImage, generateTextStream } from './api'
import './App.css'

const BUBBLE_LABELS = { user: 'You', assistant: 'Model', analysis: 'Image analysis' }

function MessageBubble({ role, text }) {
  return (
    <div className={`bubble ${role}`}>
      <span className="bubble-label">{BUBBLE_LABELS[role] ?? role}</span>
      {role === 'analysis' ? <pre>{text}</pre> : <p>{text}</p>}
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

  async function handlePaste(event) {
    if (loading) return  // the input is disabled while loading, but guard
    // directly too -- matches handleSend's own `loading` check below, and
    // stays correct even if a paste target other than this input is ever added
    const item = Array.from(event.clipboardData.items).find((clipboardItem) => clipboardItem.type.startsWith('image/'))
    if (!item) return  // plain text paste -- let the browser handle it normally
    event.preventDefault()

    const file = item.getAsFile()
    setMessages((prev) => [...prev, { role: 'user', text: `📷 pasted image (${file.type})` }])
    setLoading(true)
    setError(null)

    try {
      const result = await analyzeImage(file)
      setMessages((prev) => [...prev, { role: 'analysis', text: JSON.stringify(result, null, 2) }])
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

  const lastMessage = messages[messages.length - 1]
  const isAwaitingFirstChunk = loading && lastMessage?.role === 'assistant' && lastMessage.text === ''

  return (
    <div className="chat-app">
      <header>
        <div>
          <h1>GPTbot</h1>
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
          onPaste={handlePaste}
          placeholder="Type a prompt, or paste an image to analyze…"
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
