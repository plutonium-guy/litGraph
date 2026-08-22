//! Gateway server and virtual-key generator.

use std::sync::Arc;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "litgraph-gateway", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the OpenAI-compatible gateway.
    Serve {
        #[arg(long, default_value = "gateway.toml")]
        config: String,
        #[arg(long, default_value = "127.0.0.1:8080")]
        bind: String,
    },
    /// Mint a virtual API key and print its config stanza.
    Keygen {
        #[arg(long)]
        id: String,
        #[arg(long = "group", value_name = "GROUP", required = true)]
        groups: Vec<String>,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    match Cli::parse().command {
        Command::Keygen { id, groups } => {
            let (plaintext, prefix, hash) = litgraph_gateway::keys::generate_key();
            println!("# Store this key now; it is not recoverable.\n{plaintext}\n");
            println!("[[key]]");
            println!("id = {id:?}");
            println!("prefix = {prefix:?}");
            println!("hash = {hash:?}");
            println!("groups = {groups:?}");
        }
        Command::Serve { config, bind } => {
            let text = std::fs::read_to_string(&config)?;
            let cfg = litgraph_gateway::config::GatewayConfig::from_toml_str(&text)?;
            let state = Arc::new(litgraph_gateway::build_state(&cfg)?);
            let listener = tokio::net::TcpListener::bind(&bind).await?;
            let address = listener.local_addr()?;
            tracing::info!(%address, "litgraph-gateway listening");
            axum::serve(listener, litgraph_gateway::http::router(state)).await?;
        }
    }
    Ok(())
}
