package dev.alvo;

import java.util.Arrays;
import java.util.Collection;
import java.util.Locale;

/**
 * Renders a single, in-place-updating ASCII progress line for the training loop: overall
 * progress bar, iteration counts, current/average loss, ETA, and a short loss-trend sparkline.
 * Writes with \r (no trailing newline) so each call overwrites the previous render in place.
 *
 * One instance per training run, driven once per iteration from Inference#train. Not
 * thread-safe; training is single-threaded so this is not a concern. No internal throttling:
 * a single iteration's forward/backward pass is far more expensive than one stdout write/flush.
 */
public class TrainingProgressReporter {

  private static final int BAR_WIDTH = 30;
  private static final int SPARKLINE_WIDTH = 20;
  private static final char[] SPARK_LEVELS = "▁▂▃▄▅▆▇█".toCharArray();

  private final int totalIterations;
  private final long startTimeMillis;
  private int lastLineLength = 0;

  public TrainingProgressReporter(int totalIterations) {
    this.totalIterations = Math.max(totalIterations, 1);
    this.startTimeMillis = System.currentTimeMillis();
  }

  /** Call once per completed iteration; completedIterations must be &gt;= 1. */
  public void report(int completedIterations, double currentLoss, double movingAverageLoss,
                      Collection<Double> recentLossWindow) {
    var fraction = Math.min(1d, completedIterations / (double) totalIterations);
    var line = String.format(Locale.ROOT,
      "\r[%s] %5.1f%% (%d/%d) loss=%.4f avg=%.4f eta=%s trend:%s",
      renderBar(fraction), fraction * 100, completedIterations, totalIterations,
      currentLoss, movingAverageLoss, renderEta(completedIterations),
      renderSparkline(recentLossWindow));

    print(line);
  }

  /** Escapes the in-place line before multi-line output (checkpoint save + sampling). */
  public void breakLine() {
    System.out.println();
    lastLineLength = 0;
  }

  /** Call once after the training loop ends to leave the cursor/prompt on a clean line. */
  public void finish() {
    System.out.println();
    System.out.printf(Locale.ROOT, "Training complete: %d iterations in %s%n",
      totalIterations, formatDuration(System.currentTimeMillis() - startTimeMillis));
  }

  private void print(String content) {
    var padded = content.length() < lastLineLength
      ? content + " ".repeat(lastLineLength - content.length())
      : content;
    lastLineLength = content.length();
    System.out.print(padded);
    System.out.flush();
  }

  private static String renderBar(double fraction) {
    int filled = (int) Math.round(fraction * BAR_WIDTH);
    var bar = new StringBuilder(BAR_WIDTH);
    for (int i = 0; i < BAR_WIDTH; i++) {
      bar.append(i < filled ? '█' : '░');
    }
    return bar.toString();
  }

  private String renderEta(int completedIterations) {
    if (completedIterations <= 0) {
      return "--:--";
    }
    var elapsedMillis = System.currentTimeMillis() - startTimeMillis;
    var msPerIteration = elapsedMillis / (double) completedIterations;
    var remaining = Math.max(totalIterations - completedIterations, 0);
    return formatDuration((long) (msPerIteration * remaining));
  }

  private static String renderSparkline(Collection<Double> losses) {
    if (losses.isEmpty()) {
      return "";
    }
    var values = losses.stream()
      .mapToDouble(Double::doubleValue)
      .skip(Math.max(0, losses.size() - SPARKLINE_WIDTH))
      .toArray();
    var min = Arrays.stream(values).min().orElse(0);
    var max = Arrays.stream(values).max().orElse(0);
    var range = max - min;

    var sb = new StringBuilder(values.length);
    for (var v : values) {
      int level = range < 1e-9
        ? SPARK_LEVELS.length / 2
        : (int) Math.round((v - min) / range * (SPARK_LEVELS.length - 1));
      sb.append(SPARK_LEVELS[level]);
    }
    return sb.toString();
  }

  private static String formatDuration(long millis) {
    var totalSeconds = Math.max(0, millis / 1000);
    var hours = totalSeconds / 3600;
    var minutes = (totalSeconds % 3600) / 60;
    var seconds = totalSeconds % 60;
    return hours > 0
      ? String.format(Locale.ROOT, "%d:%02d:%02d", hours, minutes, seconds)
      : String.format(Locale.ROOT, "%02d:%02d", minutes, seconds);
  }
}
