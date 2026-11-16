import './globals.css';

export const metadata = {
  title: 'AquaEnhance — Victorian Underwater Image Enhancement',
  description: 'Automated Underwater Optical Enhancement System with U-Net Neural Engine',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
