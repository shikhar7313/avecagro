export const getReceiptIcon = ({ deliveredTo = [], seenBy = [], username }) => {
  const delivered = deliveredTo.includes(username);
  const seen = seenBy.includes(username);
  if (seen) return '✓✓';
  if (delivered) return '✓✓';
  return '✓';
};

export const getReceiptClass = ({ deliveredTo = [], seenBy = [], username }) => {
  const seen = seenBy.includes(username);
  const delivered = deliveredTo.includes(username);
  if (seen) return 'receipt receipt--seen';
  if (delivered) return 'receipt receipt--delivered';
  return 'receipt receipt--sent';
};
