/**
 * Feedback widget for rating assistant responses
 */
import React, { useState } from 'react';
import { ThumbsUp, ThumbsDown, MessageSquare } from 'lucide-react';
import { submitFeedback } from '../services/api';

interface FeedbackWidgetProps {
  chatId: number;
}

const FeedbackWidget: React.FC<FeedbackWidgetProps> = ({ chatId }) => {
  const [rating, setRating] = useState<number | null>(null);
  const [showComment, setShowComment] = useState(false);
  const [comment, setComment] = useState('');
  const [submitted, setSubmitted] = useState(false);

  const handleRating = async (score: number) => {
    setRating(score);
    
    if (score === 1) {
      // Positive feedback - submit immediately
      try {
        await submitFeedback(chatId, score);
        setSubmitted(true);
      } catch (error) {
        console.error('Error submitting feedback:', error);
      }
    } else {
      // Negative feedback - show comment box
      setShowComment(true);
    }
  };

  const handleSubmitComment = async () => {
    if (rating === null) return;

    try {
      await submitFeedback(chatId, rating, comment);
      setSubmitted(true);
      setShowComment(false);
    } catch (error) {
      console.error('Error submitting feedback:', error);
    }
  };

  if (submitted) {
    return (
      <div className="text-xs text-green-600">
        ✓ Thank you for your feedback!
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {/* Rating Buttons */}
      <div className="flex gap-2 items-center">
        <span className="text-xs text-gray-500">Was this helpful?</span>
        <button
          onClick={() => handleRating(1)}
          className={`p-1 rounded transition-colors ${
            rating === 1 
              ? 'text-green-600 bg-green-50' 
              : 'text-gray-400 hover:text-green-600 hover:bg-green-50'
          }`}
          title="Thumbs up"
        >
          <ThumbsUp size={16} />
        </button>
        <button
          onClick={() => handleRating(-1)}
          className={`p-1 rounded transition-colors ${
            rating === -1 
              ? 'text-red-600 bg-red-50' 
              : 'text-gray-400 hover:text-red-600 hover:bg-red-50'
          }`}
          title="Thumbs down"
        >
          <ThumbsDown size={16} />
        </button>
      </div>

      {/* Comment Box */}
      {showComment && (
        <div className="bg-gray-50 p-3 rounded-lg space-y-2">
          <div className="flex items-center gap-2 text-sm text-gray-700">
            <MessageSquare size={16} />
            <span>What went wrong?</span>
          </div>
          <textarea
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            placeholder="Optional: Tell us more about the issue..."
            className="w-full p-2 text-sm border border-gray-300 rounded resize-none focus:outline-none focus:ring-2 focus:ring-indigo-500"
            rows={3}
          />
          <div className="flex gap-2 justify-end">
            <button
              onClick={() => setShowComment(false)}
              className="px-3 py-1 text-sm text-gray-600 hover:text-gray-800"
            >
              Cancel
            </button>
            <button
              onClick={handleSubmitComment}
              className="px-3 py-1 text-sm bg-indigo-500 text-white rounded hover:bg-indigo-600"
            >
              Submit
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default FeedbackWidget;

