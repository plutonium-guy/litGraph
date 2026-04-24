//! Prompt snippets to inject into an LLM request so the response matches
//! what a given parser can consume. LangChain `get_format_instructions()`
//! parity — every output parser ships its own prompt in LangChain.
//!
//! # Pattern
//!
//! ```ignore
//! let prompt = format!(
//!     "List 5 popular fruits.\n\n{}",
//!     comma_list_format_instructions()
//! );
//! let response = chat.invoke(prompt).await?;
//! let fruits = parse_comma_list(&response.content);
//! ```
//!
//! Each helper is a pure string function — no allocations beyond the
//! returned `String`, no state. Safe to call in a hot loop (though
//! you should cache the result if you do).

/// Prompt snippet for `parse_comma_list`. Tells the LLM to produce a
/// single comma-separated line.
pub fn comma_list_format_instructions() -> String {
    "Your response should be a comma-separated list of values, \
     like this: item1, item2, item3. Do not number the items. \
     Do not include any other prose before or after the list."
        .to_string()
}

/// Prompt snippet for `parse_numbered_list`. Tells the LLM to produce
/// a `1.`-delimited list.
pub fn numbered_list_format_instructions() -> String {
    "Respond with a numbered list, one item per line, in the format:\n\
     1. first item\n\
     2. second item\n\
     3. third item\n\
     Do not include other prose around the list."
        .to_string()
}

/// Prompt snippet for `parse_markdown_list`. Tells the LLM to produce
/// a `-`-bulleted list.
pub fn markdown_list_format_instructions() -> String {
    "Respond with a markdown bulleted list, one item per line:\n\
     - first item\n\
     - second item\n\
     - third item\n\
     Do not include other prose around the list."
        .to_string()
}

/// Prompt snippet for `parse_boolean`. Tells the LLM to answer yes or no.
pub fn boolean_format_instructions() -> String {
    "Respond with a single word: yes or no. \
     Do not include any other prose."
        .to_string()
}

/// Prompt snippet for `parse_xml_tags` / `parse_nested_xml`. Tells the
/// LLM to wrap each requested field in its matching XML tag.
///
/// ```ignore
/// let prompt = format!(
///     "Solve this problem.\n\n{}",
///     xml_format_instructions(&["thinking", "answer"])
/// );
/// // Instructs: wrap reasoning in <thinking>...</thinking> and the final
/// // answer in <answer>...</answer>.
/// ```
pub fn xml_format_instructions(tags: &[&str]) -> String {
    let tag_list = tags
        .iter()
        .map(|t| format!("<{t}>...</{t}>"))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "Format your response using XML tags. Wrap each section in its \
         matching tag:\n{tag_list}\n\
         Use the tags exactly as shown. Do not include attributes. Use \
         `&lt;`, `&gt;`, `&amp;` to escape literal angle brackets and \
         ampersands inside content."
    )
}

/// Prompt snippet for `parse_react_step`. Tells the LLM to use the
/// Thought/Action/Action Input/Final Answer format with the provided
/// tool catalog.
///
/// ```ignore
/// let tool_lines = vec![
///     "get_weather: fetches the current weather for a city",
///     "web_search: searches the web for a query",
/// ];
/// let prompt = react_format_instructions(&tool_lines);
/// ```
pub fn react_format_instructions(tools: &[&str]) -> String {
    let tool_block = if tools.is_empty() {
        "(no tools available — you must respond with Final Answer only)".to_string()
    } else {
        tools
            .iter()
            .map(|t| format!("- {t}"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    format!(
        "You have access to the following tools:\n\
         {tool_block}\n\
         \n\
         Use the following format to reason:\n\
         \n\
         Thought: reason about what to do next.\n\
         Action: <exact name of one of the tools above>\n\
         Action Input: <JSON object, or a plain string, matching the tool's expected input>\n\
         Observation: (the tool's response will be provided here)\n\
         ... (Thought/Action/Action Input/Observation may repeat)\n\
         \n\
         When you have enough information to answer the user's question, respond with:\n\
         \n\
         Thought: I have the final answer.\n\
         Final Answer: <your final answer to the user>"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comma_list_instructions_mention_format() {
        let s = comma_list_format_instructions();
        assert!(s.contains("comma-separated"));
    }

    #[test]
    fn numbered_list_instructions_show_example() {
        let s = numbered_list_format_instructions();
        assert!(s.contains("1."));
        assert!(s.contains("2."));
    }

    #[test]
    fn markdown_list_instructions_show_example() {
        let s = markdown_list_format_instructions();
        assert!(s.contains("- first item"));
    }

    #[test]
    fn boolean_instructions_specify_single_word() {
        let s = boolean_format_instructions();
        assert!(s.to_lowercase().contains("yes or no"));
    }

    #[test]
    fn xml_instructions_enumerate_each_tag() {
        let s = xml_format_instructions(&["thinking", "answer"]);
        assert!(s.contains("<thinking>...</thinking>"));
        assert!(s.contains("<answer>...</answer>"));
    }

    #[test]
    fn xml_instructions_mention_entity_escapes() {
        let s = xml_format_instructions(&["x"]);
        assert!(s.contains("&lt;"));
        assert!(s.contains("&amp;"));
    }

    #[test]
    fn react_instructions_list_each_tool() {
        let s = react_format_instructions(&[
            "get_weather: fetch the weather",
            "web_search: search the web",
        ]);
        assert!(s.contains("get_weather"));
        assert!(s.contains("web_search"));
        assert!(s.contains("Thought:"));
        assert!(s.contains("Action:"));
        assert!(s.contains("Action Input:"));
        assert!(s.contains("Final Answer:"));
    }

    #[test]
    fn react_instructions_handles_no_tools_gracefully() {
        let s = react_format_instructions(&[]);
        assert!(s.to_lowercase().contains("no tools available"));
        // Still describes the Final Answer shape.
        assert!(s.contains("Final Answer:"));
    }

    #[test]
    fn outputs_round_trip_with_parser() {
        // Integration-check: an LLM that followed the instructions would
        // produce output our parser understands. Simulate with a prose
        // string that matches each instruction's example.
        use crate::{
            parse_boolean, parse_comma_list, parse_markdown_list, parse_numbered_list,
        };
        assert_eq!(
            parse_comma_list("item1, item2, item3"),
            vec!["item1", "item2", "item3"]
        );
        assert_eq!(
            parse_numbered_list("1. first item\n2. second item\n3. third item"),
            vec!["first item", "second item", "third item"]
        );
        assert_eq!(
            parse_markdown_list("- first item\n- second item\n- third item"),
            vec!["first item", "second item", "third item"]
        );
        assert_eq!(parse_boolean("yes").unwrap(), true);
        assert_eq!(parse_boolean("no").unwrap(), false);
    }
}
