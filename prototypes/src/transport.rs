use quinn::TransportConfig;
use std::convert::TryInto;
use std::sync::Arc;
use std::time::Duration;

/// Prototype QUIC transport tuning. Builds a Quinn [`TransportConfig`].
#[derive(Debug, Clone)]
pub struct TransportSettings {
    pub max_idle_timeout: Duration,
    pub keep_alive_interval: Option<Duration>,
    pub stream_receive_window: u32,
    pub receive_window: u32,
    pub send_window: u64,
    pub max_concurrent_bidi_streams: u32,
    pub max_concurrent_uni_streams: u32,
}

impl Default for TransportSettings {
    fn default() -> Self {
        Self {
            max_idle_timeout: Duration::from_secs(3),
            keep_alive_interval: Some(Duration::from_secs(5)),
            stream_receive_window: 2 * 1024 * 1024,
            receive_window: 6 * 1024 * 1024,
            send_window: 2 * 1024 * 1024,
            max_concurrent_bidi_streams: 32,
            max_concurrent_uni_streams: 32,
        }
    }
}

impl TransportSettings {
    pub fn build(&self) -> Result<Arc<TransportConfig>, quinn::VarIntBoundsExceeded> {
        let mut config = TransportConfig::default();
        config.max_idle_timeout(Some(self.max_idle_timeout.try_into()?));
        config.keep_alive_interval(self.keep_alive_interval);
        config.stream_receive_window(self.stream_receive_window.into());
        config.receive_window(self.receive_window.into());
        config.send_window(self.send_window);
        config.max_concurrent_bidi_streams(self.max_concurrent_bidi_streams.into());
        config.max_concurrent_uni_streams(self.max_concurrent_uni_streams.into());
        Ok(Arc::new(config))
    }
}
