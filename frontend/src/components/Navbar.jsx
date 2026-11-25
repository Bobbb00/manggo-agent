import { useState } from 'react'
import { Menu, X, Leaf } from 'lucide-react'

const navLinks = [
  { href: '#', label: 'Beranda' },
  { href: '#klasifikasi', label: 'Klasifikasi' },
  { href: '#riwayat', label: 'Riwayat' },
]

export default function Navbar() {
  const [mobileOpen, setMobileOpen] = useState(false)

  const closeMenu = () => setMobileOpen(false)

  return (
    <>
      <header className="bg-green-600 text-white shadow-md sticky top-0 z-50">
        <nav className="container mx-auto flex justify-between items-center px-4 py-3.5">
          <a href="#" className="flex items-center gap-2 text-xl font-bold tracking-tight">
            <Leaf className="w-6 h-6 text-yellow-200" />
            <span>ManggaAI</span>
          </a>

          <ul className="hidden md:flex space-x-6 text-sm font-medium">
            {navLinks.map((link) => (
              <li key={link.label}>
                <a href={link.href} className="hover:text-yellow-200 transition-colors">
                  {link.label}
                </a>
              </li>
            ))}
          </ul>

          <button
            type="button"
            className="md:hidden p-1.5 rounded-lg hover:bg-green-700 focus:outline-none"
            onClick={() => setMobileOpen(true)}
            aria-label="Buka menu navigasi"
          >
            <Menu className="w-6 h-6" />
          </button>
        </nav>
      </header>

      {/* Mobile Backdrop */}
      {mobileOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 md:hidden"
          onClick={closeMenu}
          aria-hidden="true"
        />
      )}

      {/* Mobile Drawer */}
      <div
        className={`fixed inset-y-0 right-0 w-64 bg-green-700 text-white shadow-xl transform transition-transform duration-300 md:hidden z-50 p-5 ${
          mobileOpen ? 'translate-x-0' : 'translate-x-full'
        }`}
        aria-hidden={!mobileOpen}
      >
        <div className="flex justify-between items-center mb-6">
          <span className="font-bold text-lg">Menu</span>
          <button
            type="button"
            className="p-1 rounded-lg hover:bg-green-600"
            onClick={closeMenu}
            aria-label="Tutup menu"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        <ul className="flex flex-col space-y-4 text-sm font-medium">
          {navLinks.map((link) => (
            <li key={link.label}>
              <a
                href={link.href}
                className="block py-1 hover:text-yellow-200 transition-colors"
                onClick={closeMenu}
              >
                {link.label}
              </a>
            </li>
          ))}
        </ul>
      </div>
    </>
  )
}
